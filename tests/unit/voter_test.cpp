#include <gtest/gtest.h>

#include "tasks/auto_aim/voter.hpp"

TEST(Voter, InitialCountsZero)
{
  auto_aim::Voter voter;
  EXPECT_EQ(voter.count(auto_aim::red, auto_aim::one, auto_aim::small), 0u);
  EXPECT_EQ(voter.count(auto_aim::blue, auto_aim::sentry, auto_aim::big), 0u);
}

TEST(Voter, SingleVote)
{
  auto_aim::Voter voter;
  voter.vote(auto_aim::red, auto_aim::three, auto_aim::small);
  EXPECT_EQ(voter.count(auto_aim::red, auto_aim::three, auto_aim::small), 1u);
}

TEST(Voter, MultipleVotes)
{
  auto_aim::Voter voter;
  for (int i = 0; i < 10; i++) {
    voter.vote(auto_aim::blue, auto_aim::four, auto_aim::big);
  }
  EXPECT_EQ(voter.count(auto_aim::blue, auto_aim::four, auto_aim::big), 10u);
}

TEST(Voter, IndependentBuckets)
{
  auto_aim::Voter voter;
  voter.vote(auto_aim::red, auto_aim::one, auto_aim::small);
  voter.vote(auto_aim::red, auto_aim::one, auto_aim::small);
  voter.vote(auto_aim::blue, auto_aim::one, auto_aim::small);

  EXPECT_EQ(voter.count(auto_aim::red, auto_aim::one, auto_aim::small), 2u);
  EXPECT_EQ(voter.count(auto_aim::blue, auto_aim::one, auto_aim::small), 1u);
  EXPECT_EQ(voter.count(auto_aim::extinguish, auto_aim::one, auto_aim::small), 0u);
}

TEST(Voter, AllColors)
{
  auto_aim::Voter voter;
  voter.vote(auto_aim::red, auto_aim::sentry, auto_aim::small);
  voter.vote(auto_aim::blue, auto_aim::sentry, auto_aim::small);
  voter.vote(auto_aim::extinguish, auto_aim::sentry, auto_aim::small);
  voter.vote(auto_aim::purple, auto_aim::sentry, auto_aim::small);

  EXPECT_EQ(voter.count(auto_aim::red, auto_aim::sentry, auto_aim::small), 1u);
  EXPECT_EQ(voter.count(auto_aim::blue, auto_aim::sentry, auto_aim::small), 1u);
  EXPECT_EQ(voter.count(auto_aim::extinguish, auto_aim::sentry, auto_aim::small), 1u);
  EXPECT_EQ(voter.count(auto_aim::purple, auto_aim::sentry, auto_aim::small), 1u);
}

TEST(Voter, AllArmorTypes)
{
  auto_aim::Voter voter;
  voter.vote(auto_aim::red, auto_aim::three, auto_aim::small);
  voter.vote(auto_aim::red, auto_aim::three, auto_aim::big);

  EXPECT_EQ(voter.count(auto_aim::red, auto_aim::three, auto_aim::small), 1u);
  EXPECT_EQ(voter.count(auto_aim::red, auto_aim::three, auto_aim::big), 1u);
}

TEST(Voter, DifferentNames)
{
  auto_aim::Voter voter;
  voter.vote(auto_aim::red, auto_aim::one, auto_aim::small);
  voter.vote(auto_aim::red, auto_aim::two, auto_aim::small);
  voter.vote(auto_aim::red, auto_aim::three, auto_aim::small);
  voter.vote(auto_aim::red, auto_aim::outpost, auto_aim::small);
  voter.vote(auto_aim::red, auto_aim::base, auto_aim::big);

  EXPECT_EQ(voter.count(auto_aim::red, auto_aim::one, auto_aim::small), 1u);
  EXPECT_EQ(voter.count(auto_aim::red, auto_aim::two, auto_aim::small), 1u);
  EXPECT_EQ(voter.count(auto_aim::red, auto_aim::three, auto_aim::small), 1u);
  EXPECT_EQ(voter.count(auto_aim::red, auto_aim::outpost, auto_aim::small), 1u);
  EXPECT_EQ(voter.count(auto_aim::red, auto_aim::base, auto_aim::big), 1u);
  EXPECT_EQ(voter.count(auto_aim::red, auto_aim::four, auto_aim::small), 0u);
}
