#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

class ControllerNodeCpp : public rclcpp::Node
{
public:
  ControllerNodeCpp()
  : Node("controller_node_cpp")
  {
    k_ext_ = this->declare_parameter<std::vector<double>>(
      "controller.k_ext", {-1.8755, -29.2582, -0.9120, 0.4696});

    if (k_ext_.size() != 4) {
      RCLCPP_WARN(this->get_logger(), "controller.k_ext must have 4 values. Falling back to defaults.");
      k_ext_ = {-1.8755, -29.2582, -0.9120, 0.4696};
    }

    ts_ = this->declare_parameter<double>("controller.ts", 0.005);
    phi_max_deg_ = this->declare_parameter<double>("controller.phi_max_deg", 8.0);
    u_max_ = this->declare_parameter<double>("controller.u_max", 0.2);
    x_err_max_ = this->declare_parameter<double>("controller.x_err_max", 0.5);
    s_dot_ref_ = this->declare_parameter<double>("controller.s_dot_ref", 0.0);
    state_topic_ = this->declare_parameter<std::string>("topics.state_vector", "/state_vector");
    cmd_topic_ = this->declare_parameter<std::string>("topics.cmd_vel", "/cmd_vel");

    enable_timing_stats_ = this->declare_parameter<bool>("timing.enable_stats", true);
    timing_log_interval_s_ = this->declare_parameter<double>("timing.log_interval_s", 5.0);
    warn_jitter_ms_ = this->declare_parameter<double>("timing.warn_jitter_ms", 1.0);

    const double phi_max_rad = phi_max_deg_ * M_PI / 180.0;
    phi_max_ = phi_max_rad;

    state_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      state_topic_, 10,
      std::bind(&ControllerNodeCpp::state_callback, this, std::placeholders::_1));

    cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_topic_, 10);

    auto period = std::chrono::duration<double>(ts_);
    control_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&ControllerNodeCpp::control_loop, this));

    RCLCPP_INFO(
      this->get_logger(),
      "Controller C++ Node started. K=[%.4f %.4f %.4f %.4f], phi_max=%.1f deg, u_max=%.3f m/s",
      k_ext_[0], k_ext_[1], k_ext_[2], k_ext_[3], phi_max_deg_, u_max_);

    if (enable_timing_stats_) {
      RCLCPP_INFO(
        this->get_logger(),
        "Timing stats enabled: Ts=%.3f ms, log_interval=%.1f s, warn_jitter>%.3f ms",
        ts_ * 1000.0, timing_log_interval_s_, warn_jitter_ms_);
    }
  }

private:
  void state_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->data.size() < 4) {
      RCLCPP_WARN(this->get_logger(), "Invalid /state_vector size=%zu. Need at least 4 values.", msg->data.size());
      return;
    }

    s_dot_ = msg->data[1];
    phi_ = msg->data[2];
    phi_dot_ = msg->data[3];
    state_received_ = true;
  }

  void control_loop()
  {
    update_timing_stats();

    if (!state_received_) {
      return;
    }

    if (std::abs(phi_) > phi_max_) {
      if (active_) {
        RCLCPP_WARN(
          this->get_logger(),
          "phi=%.2f deg > %.2f deg -> controller disabled",
          phi_ * 180.0 / M_PI, phi_max_deg_);
      }
      active_ = false;
      x_err_ = 0.0;
      publish_cmd(0.0);
      return;
    }

    active_ = true;

    x_err_ += (s_dot_ - s_dot_ref_) * ts_;
    x_err_ = std::clamp(x_err_, -x_err_max_, x_err_max_);

    std::array<double, 4> x{
      s_dot_ - s_dot_ref_,
      phi_,
      phi_dot_,
      x_err_};

    double u = -(
      k_ext_[0] * x[0] +
      k_ext_[1] * x[1] +
      k_ext_[2] * x[2] +
      k_ext_[3] * x[3]);

    u = std::clamp(u, -u_max_, u_max_);
    publish_cmd(u);
  }

  void publish_cmd(double v)
  {
    geometry_msgs::msg::Twist msg;
    msg.linear.x = v;
    cmd_pub_->publish(msg);
  }

  void reset_timing_window(const std::chrono::steady_clock::time_point & now)
  {
    window_start_ = now;
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
    if (!enable_timing_stats_) {
      return;
    }

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

    if (abs_jitter * 1000.0 > warn_jitter_ms_) {
      ++jitter_warn_count_;
    }

    if (dt > (1.5 * ts_)) {
      ++overrun_count_;
    }

    if (!next_log_.has_value()) {
      reset_timing_window(now);
      return;
    }

    if (now >= *next_log_ && dt_count_ > 0) {
      const double avg_dt = dt_sum_ / static_cast<double>(dt_count_);
      RCLCPP_INFO(
        this->get_logger(),
        "Timing: n=%zu, avg=%.3f ms, min=%.3f ms, max=%.3f ms, max|jitter|=%.3f ms, jitter_warns=%zu, overruns=%zu",
        dt_count_, avg_dt * 1000.0, dt_min_ * 1000.0, dt_max_ * 1000.0,
        max_abs_jitter_ * 1000.0, jitter_warn_count_, overrun_count_);
      reset_timing_window(now);
    }
  }

  std::vector<double> k_ext_;
  double ts_{0.005};
  double phi_max_deg_{8.0};
  double phi_max_{8.0 * M_PI / 180.0};
  double u_max_{0.2};
  double x_err_max_{0.5};
  double s_dot_ref_{0.0};

  std::string state_topic_{"/state_vector"};
  std::string cmd_topic_{"/cmd_vel"};

  double s_dot_{0.0};
  double phi_{0.0};
  double phi_dot_{0.0};
  double x_err_{0.0};
  bool state_received_{false};
  bool active_{false};

  bool enable_timing_stats_{true};
  double timing_log_interval_s_{5.0};
  double warn_jitter_ms_{1.0};

  std::optional<std::chrono::steady_clock::time_point> last_loop_;
  std::optional<std::chrono::steady_clock::time_point> window_start_;
  std::optional<std::chrono::steady_clock::time_point> next_log_;
  std::size_t dt_count_{0};
  double dt_sum_{0.0};
  double dt_min_{0.0};
  double dt_max_{0.0};
  double max_abs_jitter_{0.0};
  std::size_t jitter_warn_count_{0};
  std::size_t overrun_count_{0};

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr state_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ControllerNodeCpp>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
