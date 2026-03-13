#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/multi_array_dimension.hpp"
#include "turtlebot3_node/msg/wheel_pwm.hpp"

namespace
{
constexpr int kPwmLimit = 885;
}

// ---------------------------------------------------------------------------
// Diskreter Tiefpass 1. Ordnung (Tustin)
// ---------------------------------------------------------------------------
class LowPassFilter
{
public:
  LowPassFilter(double f_grenz, double ts)
  {
    const double omega_c = 2.0 * M_PI * f_grenz;
    const double k = omega_c * ts;
    b0_ = k / (2.0 + k);
    b1_ = k / (2.0 + k);
    a1_ = -(2.0 - k) / (2.0 + k);
  }

  double update(double x)
  {
    const double y = b0_ * x + b1_ * x_prev_ - a1_ * y_prev_;
    x_prev_ = x;
    y_prev_ = y;
    return y;
  }

private:
  double b0_{0.0}, b1_{0.0}, a1_{0.0};
  double x_prev_{0.0}, y_prev_{0.0};
};

// ---------------------------------------------------------------------------
// Segway Node (PWM-Ausgang)
//
// Subscriber:  /imu, /odom
// Publisher:   /cmd_pwm  (turtlebot3_node/msg/WheelPwm, beide gleich)
//              /state_vector  (Float64MultiArray, optional fuer Debug)
//
// Mapping:  u ∈ [-u_max, u_max]  →  pwm ∈ [-885, 885]  (linear)
// x = [s_dot_filtered - s_dot_ref, phi, phi_dot, x_err]
// u = -K_ext @ x
// ---------------------------------------------------------------------------
class SegwayNodePwm : public rclcpp::Node
{
public:
  SegwayNodePwm()
  : Node("segway_node_pwm")
  {
    // --- Sensor / Filter ---
    ts_ = this->declare_parameter<double>("ts", 0.005);
    f_grenz_ = this->declare_parameter<double>("f_grenz", 10.0);

    // --- Regler ---
    k_ext_ = this->declare_parameter<std::vector<double>>(
      "k_ext", {-1.8755, -29.2582, -0.9120, 0.4696});
    if (k_ext_.size() != 4) {
      RCLCPP_WARN(this->get_logger(), "k_ext muss 4 Werte haben. Nutze Standardwerte.");
      k_ext_ = {-1.8755, -29.2582, -0.9120, 0.4696};
    }

    phi_max_ = this->declare_parameter<double>("phi_max_deg", 8.0) * M_PI / 180.0;
    u_max_ = this->declare_parameter<double>("u_max", 0.2);
    x_err_max_ = this->declare_parameter<double>("x_err_max", 0.5);
    s_dot_ref_ = this->declare_parameter<double>("s_dot_ref", 0.0);

    // --- PWM ---
    pwm_max_ = this->declare_parameter<int>("pwm_max", 885);
    pwm_max_ = std::clamp(pwm_max_, 0, kPwmLimit);

    // --- Topics ---
    const auto imu_topic = this->declare_parameter<std::string>("imu_topic", "/imu");
    const auto odom_topic = this->declare_parameter<std::string>("odom_topic", "/odom");
    const auto cmd_topic = this->declare_parameter<std::string>("cmd_topic", "/cmd_pwm");
    publish_state_vec_ = this->declare_parameter<bool>("publish_state_vector", false);

    // --- Timing-Stats ---
    enable_timing_stats_ = this->declare_parameter<bool>("timing_enable", true);
    timing_log_interval_s_ = this->declare_parameter<double>("timing_log_interval_s", 5.0);
    warn_jitter_ms_ = this->declare_parameter<double>("timing_warn_jitter_ms", 1.0);

    // --- Filter ---
    lp_filter_ = std::make_unique<LowPassFilter>(f_grenz_, ts_);

    // --- Subscriber ---
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, 10,
      std::bind(&SegwayNodePwm::imu_callback, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic, 10,
      std::bind(&SegwayNodePwm::odom_callback, this, std::placeholders::_1));

    // --- Publisher ---
    cmd_pub_ = this->create_publisher<turtlebot3_node::msg::WheelPwm>(cmd_topic, 10);

    if (publish_state_vec_) {
      state_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/state_vector", 10);
    }

    // --- Regler-Timer: 200 Hz ---
    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(ts_));
    timer_ = this->create_wall_timer(period, std::bind(&SegwayNodePwm::control_loop, this));

    RCLCPP_INFO(
      this->get_logger(),
      "Segway PWM Node gestartet. K=[%.4f %.4f %.4f %.4f], "
      "phi_max=%.1f deg, u_max=%.3f, pwm_max=%d, Ts=%.3f ms",
      k_ext_[0], k_ext_[1], k_ext_[2], k_ext_[3],
      phi_max_ * 180.0 / M_PI, u_max_, pwm_max_, ts_ * 1000.0);
  }

private:
  // --------------------------------------------------------------------------
  static double quaternion_to_phi(double x, double y, double z, double w)
  {
    return std::asin(std::clamp(2.0 * (w * y - z * x), -1.0, 1.0));
  }

  // --------------------------------------------------------------------------
  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    const auto & q = msg->orientation;
    phi_ = quaternion_to_phi(q.x, q.y, q.z, q.w);
    phi_dot_ = msg->angular_velocity.y;
    imu_received_ = true;
  }

  // --------------------------------------------------------------------------
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    s_dot_raw_ = msg->twist.twist.linear.x;
    s_dot_ = lp_filter_->update(s_dot_raw_);
    odom_received_ = true;
  }

  // --------------------------------------------------------------------------
  void control_loop()
  {
    update_timing_stats();

    if (!(imu_received_ && odom_received_)) {
      return;
    }

    // Optional: /state_vector fuer Debug
    if (publish_state_vec_ && state_pub_) {
      std_msgs::msg::Float64MultiArray sv;
      std_msgs::msg::MultiArrayDimension dim;
      dim.label = "s_dot_raw,s_dot_filtered,phi,phi_dot";
      dim.size = 4;
      dim.stride = 4;
      sv.layout.dim.push_back(dim);
      sv.data = {s_dot_raw_, s_dot_, phi_, phi_dot_};
      state_pub_->publish(sv);
    }

    // Sicherheitscheck
    if (std::abs(phi_) > phi_max_) {
      if (active_) {
        RCLCPP_WARN(
          this->get_logger(),
          "phi=%.2f deg > %.2f deg -> Regler deaktiviert",
          phi_ * 180.0 / M_PI, phi_max_ * 180.0 / M_PI);
      }
      active_ = false;
      x_err_ = 0.0;
      publish_pwm(0);
      return;
    }

    active_ = true;

    // Integrator mit Anti-Windup
    x_err_ += (s_dot_ - s_dot_ref_) * ts_;
    x_err_ = std::clamp(x_err_, -x_err_max_, x_err_max_);

    // Zustandsvektor: [s_dot_err, phi, phi_dot, x_err]
    const std::array<double, 4> x{
      s_dot_ - s_dot_ref_,
      phi_,
      phi_dot_,
      x_err_};

    // Stellgroesse [-u_max, u_max]
    double u = -(k_ext_[0] * x[0] + k_ext_[1] * x[1] + k_ext_[2] * x[2] + k_ext_[3] * x[3]);
    u = std::clamp(u, -u_max_, u_max_);

    int32_t pwm = 0;
    if (std::abs(u_max_) > 1e-9) {
      // Linear auf PWM skalieren: u/u_max * pwm_max, gerundet
      pwm = static_cast<int32_t>(std::round(u / u_max_ * static_cast<double>(pwm_max_)));
    }
    pwm = std::clamp(pwm, -kPwmLimit, kPwmLimit);

    publish_pwm(pwm);
  }

  // --------------------------------------------------------------------------
  // Beide Motoren bekommen denselben PWM-Wert
  void publish_pwm(int32_t pwm)
  {
    pwm = std::clamp(pwm, -kPwmLimit, kPwmLimit);
    turtlebot3_node::msg::WheelPwm msg;
    msg.left_pwm  = static_cast<int16_t>(pwm);
    msg.right_pwm = static_cast<int16_t>(pwm);
    cmd_pub_->publish(msg);
  }

  // --------------------------------------------------------------------------
  void reset_timing_window(const std::chrono::steady_clock::time_point & now)
  {
    const auto log_interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(timing_log_interval_s_));
    next_log_ = now + log_interval;
    dt_count_ = 0;
    dt_sum_ = 0.0;
    dt_min_ = std::numeric_limits<double>::infinity();
    dt_max_ = 0.0;
    max_abs_jitter_ = 0.0;
    jitter_warn_count_ = 0;
    overrun_count_ = 0;
  }

  void update_timing_stats()
  {
    if (!enable_timing_stats_) { return; }
    const auto now = std::chrono::steady_clock::now();
    if (!last_loop_.has_value()) {
      last_loop_ = now;
      reset_timing_window(now);
      return;
    }
    const double dt = std::chrono::duration<double>(now - *last_loop_).count();
    last_loop_ = now;
    const double abs_jitter = std::abs(dt - ts_);
    ++dt_count_;
    dt_sum_ += dt;
    dt_min_ = std::min(dt_min_, dt);
    dt_max_ = std::max(dt_max_, dt);
    max_abs_jitter_ = std::max(max_abs_jitter_, abs_jitter);
    if (abs_jitter * 1000.0 > warn_jitter_ms_) { ++jitter_warn_count_; }
    if (dt > 1.5 * ts_) { ++overrun_count_; }
    if (next_log_.has_value() && now >= *next_log_ && dt_count_ > 0) {
      RCLCPP_INFO(
        this->get_logger(),
        "Timing: n=%zu, avg=%.3f ms, min=%.3f ms, max=%.3f ms, "
        "max|jitter|=%.3f ms, warns=%zu, overruns=%zu",
        dt_count_, (dt_sum_ / dt_count_) * 1000.0,
        dt_min_ * 1000.0, dt_max_ * 1000.0,
        max_abs_jitter_ * 1000.0, jitter_warn_count_, overrun_count_);
      reset_timing_window(now);
    }
  }

  // --- Parameter ---
  double ts_{0.005};
  double f_grenz_{10.0};
  std::vector<double> k_ext_;
  double phi_max_{8.0 * M_PI / 180.0};
  double u_max_{0.2};
  double x_err_max_{0.5};
  double s_dot_ref_{0.0};
  int pwm_max_{885};
  bool publish_state_vec_{false};

  // --- Sensorwerte ---
  double s_dot_raw_{0.0};
  double s_dot_{0.0};
  double phi_{0.0};
  double phi_dot_{0.0};
  bool imu_received_{false};
  bool odom_received_{false};

  // --- Regler-Zustand ---
  double x_err_{0.0};
  bool active_{false};

  // --- Timing-Stats ---
  bool enable_timing_stats_{true};
  double timing_log_interval_s_{5.0};
  double warn_jitter_ms_{1.0};
  std::optional<std::chrono::steady_clock::time_point> last_loop_;
  std::optional<std::chrono::steady_clock::time_point> next_log_;
  std::size_t dt_count_{0};
  double dt_sum_{0.0};
  double dt_min_{0.0};
  double dt_max_{0.0};
  double max_abs_jitter_{0.0};
  std::size_t jitter_warn_count_{0};
  std::size_t overrun_count_{0};

  // --- ROS-Handles ---
  std::unique_ptr<LowPassFilter> lp_filter_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<turtlebot3_node::msg::WheelPwm>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr state_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

// ---------------------------------------------------------------------------
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SegwayNodePwm>());
  rclcpp::shutdown();
  return 0;
}
