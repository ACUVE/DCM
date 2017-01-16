#pragma once

#include <utility>
#include <tuple>

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#define FORCE_INLINE __forceinline
#elif defined(__INTEL_COMPILER)
#define FORCE_INLINE inline
#elif defined(__GNUC__)
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
#define FORCE_INLINE inline
#endif

namespace meta
{
	template< std::size_t num, typename Type >
	struct multi_tuple
	{
		using type =
			decltype(
				std::tuple_cat(
					std::declval<
						typename multi_tuple<
							num - 1u,
							Type
						>::type
					>(),
					std::declval<
						std::tuple<
							Type
						>
					>()
				)
			)
		;
	};
	template< typename Type >
	struct multi_tuple< 1u, Type >
	{
		using type = std::tuple< Type >;
	};

	template< std::size_t num, typename Type >
	using multi_tuple_t = typename multi_tuple< num, Type >::type;

	namespace detail
	{
		template< std::size_t INDEX, typename Tuple1, typename Tuple2, typename Func >
		FORCE_INLINE
		std::enable_if_t<
			(std::tuple_size< Tuple1 >::value <= INDEX)
		> for_each_impl( Tuple1 const &t1, Tuple2 const &t2, Func &f )
		{
			// do nothing
		}
		template< std::size_t INDEX, typename Tuple1, typename Tuple2, typename Func >
		FORCE_INLINE
		std::enable_if_t<
			(std::tuple_size< Tuple1 >::value > INDEX)
		> for_each_impl( Tuple1 const &t1, Tuple2 const &t2, Func &f )
		{
			f( std::get< INDEX >( t1 ), std::get< INDEX >( t2 ) );
			for_each_impl< INDEX + 1u, Tuple1, Tuple2, Func >( t1, t2, f );
		}
	}
	template< typename Tuple1, typename Tuple2, typename Func >
	FORCE_INLINE
	std::enable_if_t<
		std::tuple_size< Tuple1 >::value == std::tuple_size< Tuple2 >::value &&
		(std::tuple_size< Tuple1 >::value > 0)
	> for_each( Tuple1 const &t1, Tuple2 const &t2, Func f )
	{
		detail::for_each_impl< 0u >( t1, t2, f );
	}

	namespace detail
	{
		template< std::size_t INDEX, typename Tuple1, typename Func >
		FORCE_INLINE
		std::enable_if_t<
			(std::tuple_size< Tuple1 >::value <= INDEX)
		> for_each_impl( Tuple1 const &t1, Func &f )
		{
			// do nothing
		}
		template< std::size_t INDEX, typename Tuple1, typename Func >
		FORCE_INLINE
		std::enable_if_t<
			(std::tuple_size< Tuple1 >::value > INDEX)
		> for_each_impl( Tuple1 const &t1, Func &f )
		{
			f( std::get< INDEX >( t1 ) );
			for_each_impl< INDEX + 1u, Tuple1, Func >( t1, f );
		}
	}
	template< typename Tuple1, typename Func >
	FORCE_INLINE
	std::enable_if_t<
		(std::tuple_size< Tuple1 >::value > 0)
	> for_each( Tuple1 const &t1, Func f )
	{
		detail::for_each_impl< 0u >( t1, f );
	}

	template< typename Bool, typename... Rest >
	FORCE_INLINE
	std::enable_if_t<
		std::is_same< Bool, bool >::value, bool
	> _and( Bool b, Rest... rest )
	{
		return b && _and( rest... );
	}
	template< typename T = void >
	FORCE_INLINE
	bool _and( void )
	{
		return true;
	}

	template< typename... Type >
	FORCE_INLINE
	void nothing( Type &&... )
	{
	}
}
