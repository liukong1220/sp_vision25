#include <gtest/gtest.h>

#include <cmath>

#include "tools/trajectory.hpp"

TEST(Trajectory, FlatShot)
{
  // Shooting horizontally at a target on the same level, 5m away, v0 = 30 m/s
  tools::Trajectory traj(30.0, 5.0, 0.0);
  EXPECT_FALSE(traj.unsolvable);
  EXPECT_GT(traj.fly_time, 0.0);
  // Pitch should be slightly positive (compensate for gravity)
  EXPECT_GT(traj.pitch, 0.0);
  EXPECT_LT(traj.pitch, M_PI / 4.0);
}

TEST(Trajectory, ElevatedTarget)
{
  // Target 3m away, 1m above
  tools::Trajectory traj(30.0, 3.0, 1.0);
  EXPECT_FALSE(traj.unsolvable);
  EXPECT_GT(traj.pitch, 0.0);
  EXPECT_GT(traj.fly_time, 0.0);
}

TEST(Trajectory, LowTarget)
{
  // Target 5m away, 1m below
  tools::Trajectory traj(30.0, 5.0, -1.0);
  EXPECT_FALSE(traj.unsolvable);
  EXPECT_GT(traj.fly_time, 0.0);
}

TEST(Trajectory, UnsolvableCase)
{
  // Very slow bullet, very far target → unsolvable
  tools::Trajectory traj(1.0, 1000.0, 0.0);
  EXPECT_TRUE(traj.unsolvable);
}

TEST(Trajectory, HighSpeedShortRange)
{
  // Very fast bullet, short range → near-zero pitch
  tools::Trajectory traj(100.0, 1.0, 0.0);
  EXPECT_FALSE(traj.unsolvable);
  EXPECT_NEAR(traj.pitch, 0.0, 0.01);  // nearly flat
  EXPECT_NEAR(traj.fly_time, 0.01, 0.01);
}

TEST(Trajectory, FlyTimeIsReasonable)
{
  // 5m away, v0 = 25 m/s → fly_time ≈ 0.2s
  tools::Trajectory traj(25.0, 5.0, 0.0);
  EXPECT_FALSE(traj.unsolvable);
  EXPECT_GT(traj.fly_time, 0.15);
  EXPECT_LT(traj.fly_time, 0.5);
}

TEST(Trajectory, SymmetryAboutHorizontal)
{
  // For same distance, target above vs below should yield different pitches
  tools::Trajectory traj_up(30.0, 5.0, 1.0);
  tools::Trajectory traj_down(30.0, 5.0, -1.0);
  EXPECT_FALSE(traj_up.unsolvable);
  EXPECT_FALSE(traj_down.unsolvable);
  EXPECT_GT(traj_up.pitch, traj_down.pitch);
}
