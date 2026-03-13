#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>

#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/multi_array_dimension.hpp"

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
  double b0_{0.0};
  double b1_{0.0};
  double a1_{0.0};
  double x_prev_{0.0};
  double y_prev_{0.0};
};

class StateVectorNodeCpp : public rclcpp::Node
{
public:
  StateVectorNodeCpp()
  : Node("state_vector_node_cpp")
  {
    ts_ = this->declare_parameter<double>("state.ts", 0.005);
    f_grenz_ = this->declare_parameter<double>("state.f_grenz", 10.0);

    imu_topic_ = this->declare_parameter<std::string>("topics.imu", "/imu");
    odom_topic_ = this->declare_parameter<std::string>("topics.odom", "/odom");
    state_topic_ = this->declare_parameter<std::string>("topics.state_vector", "/state_vector");

    lp_filter_ = std::make_unique<LowPassFilter>(f_grenz_, ts_);

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, 10,
      std::bind(&StateVectorNodeCpp::imu_callback, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 10,
      std::bind(&StateVectorNodeCpp::odom_callback, this, std::placeholders::_1));

    state_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(state_topic_, 10);

    auto period = std::chrono::duration<double>(ts_);
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&StateVectorNodeCpp::publish_state, this));

    RCLCPP_INFO(
      this->get_logger(),
      "State Vector C++ Node started. Ts=%.3f ms, f_grenz=%.1f Hz",
      ts_ * 1000.0, f_grenz_);
  }

private:
  static double quaternion_to_phi(double x, double y, double z, double w)
  {
    const double sin_pitch = 2.0 * (w * y - z * x);
    return std::asin(std::clamp(sin_pitch, -1.0, 1.0));
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    const auto & q = msg->orientation;
    phi_ = quaternion_to_phi(q.x, q.y, q.z, q.w);
    phi_dot_ = msg->angular_velocity.y;
    imu_received_ = true;
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    s_dot_ = msg->twist.twist.linear.x;
    s_dot_filtered_ = lp_filter_->update(s_dot_);
    odom_received_ = true;
  }

  void publish_state()
  {
    if (!(imu_received_ && odom_received_)) {
      return;
    }

    std_msgs::msg::Float64MultiArray msg;
    std_msgs::msg::MultiArrayDimension dim;
    dim.label = "s_dot_raw,s_dot_filtered,phi,phi_dot";
    dim.size = 4;
    dim.stride = 4;
    msg.layout.dim.push_back(dim);

    msg.data = {s_dot_, s_dot_filtered_, phi_, phi_dot_};
    state_pub_->publish(msg);
  }

  double ts_{0.005};
  double f_grenz_{10.0};

  std::string imu_topic_{"/imu"};
  std::string odom_topic_{"/odom"};
  std::string state_topic_{"/state_vector"};

  double s_dot_{0.0};
  double s_dot_filtered_{0.0};
  double phi_{0.0};
  double phi_dot_{0.0};
  bool imu_received_{false};
  bool odom_received_{false};

  std::unique_ptr<LowPassFilter> lp_filter_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr state_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StateVectorNodeCpp>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
