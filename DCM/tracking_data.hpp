#pragma once

#include <vector>
#include <atomic>
#include <cstdint>
#include <memory>

enum class tracking_state : std::uint8_t
{
	NO_TRACKING,
	BEFORE_TRACKING,
	TRACKING,
};

struct point_tracking_data
{
	std::atomic< tracking_state > state{ tracking_state::NO_TRACKING };
	std::atomic< float > x{ 0.0f }, y{ 0.0f };
	std::atomic< float > sx{ 0.0f }, sy{ 0.0f };
	std::atomic< unsigned int > age{ 0u };
};

struct tracking_data
{
	std::vector< std::unique_ptr< point_tracking_data[] > > point;
	std::vector< unsigned int > point_size;
	std::atomic< std::size_t > recognition_frame{ 0u };
	std::atomic< std::size_t > tracking_frame{ 0u };
};
