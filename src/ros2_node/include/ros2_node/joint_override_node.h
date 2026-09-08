#pragma once

#include "ros2_node/base_node.h"

#include "interface_protocol/msg/joint_override_command.hpp"
#include "joint_override_command/joint_override_command.h"

namespace ros2 {
class JointOverrideNode final : public LogicNode {
 public:
  JointOverrideNode(const std::shared_ptr<data::DataStore>& data_store, const std::string& param_tag,
                    const std::string& node_name = "joint_override_node")
      : LogicNode(data_store, param_tag, node_name) {}
  ~JointOverrideNode() = default;

 private:
  bool Init() override;

  bool CreateSubscription();
  void JointOverrideCommandCallback(const interface_protocol::msg::JointOverrideCommand::SharedPtr msg);

  rclcpp::Subscription<interface_protocol::msg::JointOverrideCommand>::SharedPtr joint_override_command_sub_;
};

REGISTER_LOGIC_NODE_TYPE(JointOverrideNode, "joint_override_node");
}  // namespace ros2
