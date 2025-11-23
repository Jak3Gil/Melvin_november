// Example scaffold file demonstrating rule syntax
// This file will be parsed, patterns injected into melvin.m, then deleted

// SAFETY_RULE(name="HIGH_TORQUE_NEAR_HUMAN",
//   context={ vision:HUMAN_LIMB, sensor:TORQUE>THRESH, motor:JOINT_ID },
//   effect={ block_motor:JOINT_ID, reward:-10 })

// Another example rule
// BEHAVIOR_RULE(name="AVOID_OBSTACLE",
//   context={ vision:OBSTACLE_DETECTED, sensor:DISTANCE<SAFE_RANGE },
//   effect={ adjust_path:CLEAR_ROUTE, reward:5 })

// Empty implementation - scaffold is just for pattern injection
void scaffold_example(void) {
    // This function body is ignored - only comments are parsed
}

