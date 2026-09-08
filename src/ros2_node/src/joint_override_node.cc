#include "ros2_node/joint_override_node.h"

#include <glog/logging.h>

namespace ros2 {

bool JointOverrideNode::Init() {
  if (!LogicNode::Init()) {
    return false;
  }

  return CreateSubscription();
}

bool JointOverrideNode::CreateSubscription() {
  if (!param_->subscribe_topics.has_value()) {
    return false;
  }
  const auto& subscribe_topics = param_->subscribe_topics.value();

  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();

  try {
    if (subscribe_topics.contains("joint_override_command")) {
      joint_override_command_sub_ = this->create_subscription<interface_protocol::msg::JointOverrideCommand>(
          subscribe_topics.at("joint_override_command"), qos,
          std::bind(&JointOverrideNode::JointOverrideCommandCallback, this, std::placeholders::_1));
      LOG(INFO) << "Create " << subscribe_topics.at("joint_override_command") << " subscription success";
    }

    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error creating subscriptions: " << e.what();
    return false;
  }
}

void JointOverrideNode::JointOverrideCommandCallback(
    const interface_protocol::msg::JointOverrideCommand::SharedPtr msg) {
  data::JointOverrideCommand command;
  command.weight = msg->weight;
  command.joint_indices = Eigen::Map<const Eigen::VectorXi>(msg->joint_indices.data(), msg->joint_indices.size());
  command.position = Eigen::Map<const Eigen::VectorXd>(msg->position.data(), msg->position.size());
  command.velocity = Eigen::Map<const Eigen::VectorXd>(msg->velocity.data(), msg->velocity.size());
  command.feed_forward_torque =
      Eigen::Map<const Eigen::VectorXd>(msg->feed_forward_torque.data(), msg->feed_forward_torque.size());
  command.torque = Eigen::Map<const Eigen::VectorXd>(msg->torque.data(), msg->torque.size());
  command.stiffness = Eigen::Map<const Eigen::VectorXd>(msg->stiffness.data(), msg->stiffness.size());
  command.damping = Eigen::Map<const Eigen::VectorXd>(msg->damping.data(), msg->damping.size());

  data_store_->joint_override_command.Set(command);
}

}  // namespace ros2
