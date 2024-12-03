#ifndef T_SKID_H
#define T_SKID_H


#include <cstdint>
#include <optional>
#include <queue> //for RRPCQ


#include "address.h"
#include "champsim.h"
#include "modules.h"
#include "msl/lru_table.h"


struct t_skid : public champsim::modules::prefetcher {
 struct addr_pred_entry {
   champsim::address ip{};
   champsim::block_number last_cl_addr{};
   champsim::block_number::difference_type last_stride{};
   int degree = 1;

   auto index() const
   {
     using namespace champsim::data::data_literals;
     return ip.slice_upper<2_b>();
   }
   auto tag() const
   {
     using namespace champsim::data::data_literals;
     return ip.slice_upper<2_b>();
   }
 };

 struct lookahead_entry {
   champsim::address address{};
   champsim::address::difference_type stride{};
   int degree = 0;
   uint64_t issue_cycle{};
 };

 struct ipt_entry {
   champsim::address trigger_pc{};
   champsim::address pf_addr{};


   auto index() const {
     using namespace champsim::data::data_literals;
     return pf_addr.slice_upper<2_b>();
   }
   auto tag() const {
     using namespace champsim::data::data_literals;
     return pf_addr.slice_upper<2_b>();
   }
 };

  //target table
 struct target_entry {
   champsim::address trigger_pc{};
   champsim::address target_pc{};
  
   auto index() const {
     using namespace champsim::data::data_literals;
     return trigger_pc.slice_upper<2_b>();
   }
   auto tag() const {
     using namespace champsim::data::data_literals;
     return trigger_pc.slice_upper<2_b>();
   }
 };


 constexpr static std::size_t APT_SETS = 1024; //r
 constexpr static std::size_t APT_WAYS = 8; //c
  constexpr static std::size_t IPT_SETS = 16;
 constexpr static std::size_t IPT_WAYS = 1; //since no ways, just 1.

 const size_t RRPCQ_SIZE = 2;  //from paper 2entry, stores 2 most resent target PC

 constexpr static std::size_t TT_SETS = 1024; //target table
 constexpr static std::size_t TT_WAYS = 8;

 constexpr static int PREFETCH_DEGREE = 3;

 std::optional<lookahead_entry> active_lookahead;


 champsim::msl::lru_table<addr_pred_entry> addr_pred_table{APT_SETS, APT_WAYS};
 champsim::msl::lru_table<ipt_entry> ipt_table{IPT_SETS, IPT_WAYS};
 std::queue<champsim::address> rrpcq;
 champsim::msl::lru_table<target_entry> target_table{TT_SETS, TT_WAYS};


public:
 using champsim::modules::prefetcher::prefetcher;


 uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                   uint32_t metadata_in);
 uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);
 void prefetcher_cycle_operate();
};


#endif
