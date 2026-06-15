#include <gtest/gtest.h>

#include "tools/crc.hpp"

// --- CRC8 ---

TEST(CRC, Crc8SingleByte)
{
  uint8_t data[] = {0x00};
  uint8_t crc = tools::get_crc8(data, 1);
  // Verify deterministic output for a known input
  EXPECT_NE(crc, 0x00);  // CRC8 of 0x00 with init 0xFF should not be 0
}

TEST(CRC, Crc8Consistency)
{
  uint8_t data[] = {0x01, 0x02, 0x03};
  uint8_t crc1 = tools::get_crc8(data, 3);
  uint8_t crc2 = tools::get_crc8(data, 3);
  EXPECT_EQ(crc1, crc2);
}

TEST(CRC, Crc8DifferentDataDifferentCrc)
{
  uint8_t data1[] = {0x01, 0x02, 0x03};
  uint8_t data2[] = {0x01, 0x02, 0x04};
  EXPECT_NE(tools::get_crc8(data1, 3), tools::get_crc8(data2, 3));
}

TEST(CRC, CheckCrc8Valid)
{
  uint8_t data[] = {0x01, 0x02, 0x03, 0x00};
  data[3] = tools::get_crc8(data, 3);
  EXPECT_TRUE(tools::check_crc8(data, 4));
}

TEST(CRC, CheckCrc8Invalid)
{
  uint8_t data[] = {0x01, 0x02, 0x03, 0x00};
  data[3] = tools::get_crc8(data, 3);
  data[3] ^= 0x01;  // corrupt CRC
  EXPECT_FALSE(tools::check_crc8(data, 4));
}

TEST(CRC, CheckCrc8DataCorrupted)
{
  uint8_t data[] = {0x01, 0x02, 0x03, 0x00};
  data[3] = tools::get_crc8(data, 3);
  data[1] ^= 0xFF;  // corrupt data
  EXPECT_FALSE(tools::check_crc8(data, 4));
}

// --- CRC16 ---

TEST(CRC, Crc16Consistency)
{
  uint8_t data[] = {0xAA, 0xBB, 0xCC, 0xDD};
  uint16_t crc1 = tools::get_crc16(data, 4);
  uint16_t crc2 = tools::get_crc16(data, 4);
  EXPECT_EQ(crc1, crc2);
}

TEST(CRC, Crc16DifferentDataDifferentCrc)
{
  uint8_t data1[] = {0x10, 0x20};
  uint8_t data2[] = {0x10, 0x21};
  EXPECT_NE(tools::get_crc16(data1, 2), tools::get_crc16(data2, 2));
}

TEST(CRC, CheckCrc16Valid)
{
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x00, 0x00};
  uint16_t crc = tools::get_crc16(data, 4);
  data[4] = crc & 0xFF;         // low byte
  data[5] = (crc >> 8) & 0xFF;  // high byte
  EXPECT_TRUE(tools::check_crc16(data, 6));
}

TEST(CRC, CheckCrc16Invalid)
{
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x00, 0x00};
  uint16_t crc = tools::get_crc16(data, 4);
  data[4] = crc & 0xFF;
  data[5] = (crc >> 8) & 0xFF;
  data[5] ^= 0x01;  // corrupt CRC high byte
  EXPECT_FALSE(tools::check_crc16(data, 6));
}

TEST(CRC, CheckCrc16DataCorrupted)
{
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x00, 0x00};
  uint16_t crc = tools::get_crc16(data, 4);
  data[4] = crc & 0xFF;
  data[5] = (crc >> 8) & 0xFF;
  data[0] ^= 0xFF;  // corrupt data
  EXPECT_FALSE(tools::check_crc16(data, 6));
}

TEST(CRC, Crc8EmptyData)
{
  uint8_t dummy = 0;
  uint8_t crc = tools::get_crc8(&dummy, 0);
  // With CRC8_INIT = 0xFF and zero-length input, the CRC should remain 0xFF
  EXPECT_EQ(crc, 0xFF);
}

TEST(CRC, Crc16EmptyData)
{
  uint8_t dummy = 0;
  uint16_t crc = tools::get_crc16(&dummy, 0);
  EXPECT_EQ(crc, 0xFFFF);
}
