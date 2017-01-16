#pragma once

#include <chrono>
#include <iostream>
#include <string>

class interval_timer
{
private:
	using clock = std::chrono::high_resolution_clock;

	clock::time_point start_time;
	std::ostream &os;
	std::string const label;
	std::string const fps;
	unsigned int count;
	unsigned int const show_count;

public:
	interval_timer( std::string _label, unsigned int const _show_count = 1000, std::string _fps = "fps", std::ostream &_os = std::cout )
		: start_time()
		, os( _os )
		, label( std::move( _label ) )
		, fps( std::move( _fps ) )
		, count( 0u )
		, show_count( _show_count )
	{
		start();
	}

	void start( void )
	{
		count = 0u;
		start_time = clock::now();
	}
	void interval( void )
	{
		if( count++ < show_count ) return;
		show();
	}
	void show( void )
	{
		auto const end_time = clock::now();
		auto const duration = end_time - start_time;
		auto const dd = std::chrono::duration< double >( duration ).count();
		os << label << ": " << dd * 1000.0 / count << " ms : " << 1 / dd * count << " " << fps << std::endl;
		start();
	}
};