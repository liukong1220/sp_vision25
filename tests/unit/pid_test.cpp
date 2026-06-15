#include <gtest/gtest.h>

#include <cmath>

#include "tools/pid.hpp"

TEST(PID, ProportionalOnly)
{
  tools::PID pid(0.01f, 1.0f, 0.0f, 0.0f, 100.0f, 100.0f);
  float out = pid.calc(10.0f, 0.0f);
  // error = 10, kp = 1 → pout = 10
  EXPECT_FLOAT_EQ(out, 10.0f);
  EXPECT_FLOAT_EQ(pid.pout, 10.0f);
  EXPECT_FLOAT_EQ(pid.iout, 0.0f);
}

TEST(PID, IntegralAccumulation)
{
  tools::PID pid(0.01f, 0.0f, 10.0f, 0.0f, 100.0f, 100.0f);
  // error = 5 each step, ki = 10, dt = 0.01 → iout += 5 * 0.01 * 10 = 0.5 per step
  pid.calc(5.0f, 0.0f);
  EXPECT_NEAR(pid.iout, 0.5f, 1e-5f);

  pid.calc(5.0f, 0.0f);
  EXPECT_NEAR(pid.iout, 1.0f, 1e-5f);
}

TEST(PID, IntegralClamping)
{
  tools::PID pid(0.01f, 0.0f, 1000.0f, 0.0f, 100.0f, 2.0f);
  // Large ki, small max_iout → iout should be clamped to 2.0
  for (int i = 0; i < 100; i++) {
    pid.calc(10.0f, 0.0f);
  }
  EXPECT_LE(pid.iout, 2.0f);
}

TEST(PID, DerivativeKick)
{
  tools::PID pid(0.01f, 0.0f, 0.0f, 0.01f, 100.0f, 100.0f);
  // First call: fdb changes from 0 to 0 → de = 0
  pid.calc(0.0f, 0.0f);
  EXPECT_FLOAT_EQ(pid.dout, 0.0f);

  // Second call: fdb changes from 0 to 10 → de = 0 - 10 = -10
  // dout = de / dt * kd = -10 / 0.01 * 0.01 = -10
  float out = pid.calc(0.0f, 10.0f);
  EXPECT_NEAR(pid.dout, -10.0f, 1e-5f);
}

TEST(PID, OutputClamping)
{
  tools::PID pid(0.01f, 100.0f, 0.0f, 0.0f, 5.0f, 5.0f);
  // error = 10, kp = 100 → pout = 1000, clamped to 5.0
  float out = pid.calc(10.0f, 0.0f);
  EXPECT_FLOAT_EQ(out, 5.0f);
}

TEST(PID, OutputClampingNegative)
{
  tools::PID pid(0.01f, 100.0f, 0.0f, 0.0f, 5.0f, 5.0f);
  float out = pid.calc(-10.0f, 0.0f);
  EXPECT_FLOAT_EQ(out, -5.0f);
}

TEST(PID, AngularModeWrapping)
{
  tools::PID pid(0.01f, 1.0f, 0.0f, 0.0f, 100.0f, 100.0f, true);
  // set = pi - 0.1, fdb = -pi + 0.1 → raw error = 2*pi - 0.2, wrapped ≈ -0.2
  float set = static_cast<float>(M_PI - 0.1);
  float fdb = static_cast<float>(-M_PI + 0.1);
  float out = pid.calc(set, fdb);
  EXPECT_NEAR(out, -0.2f, 0.01f);
}

TEST(PID, ZeroError)
{
  tools::PID pid(0.01f, 1.0f, 1.0f, 1.0f, 100.0f, 100.0f);
  float out = pid.calc(5.0f, 5.0f);
  // error = 0, no previous state → all terms 0
  EXPECT_FLOAT_EQ(pid.pout, 0.0f);
}
