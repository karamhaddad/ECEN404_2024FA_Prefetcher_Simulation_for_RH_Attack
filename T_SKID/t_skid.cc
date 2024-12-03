#include "t_skid.h"
#include "cache.h"
#include <iostream>

//new prefetches are made here, hits or misses, etc.
//this makes sure whatever i prefetch from this adr is on the same page as the prefetches i issued from the address.
uint32_t t_skid::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                         uint32_t metadata_in)
{
 champsim::block_number cl_addr{addr};
 champsim::block_number::difference_type stride = 0;


 champsim::address current_pc = ip; //current PC is the target IP regardless of hit or miss.
 //ALL the trigger PCs in the RRPCQ are read
 while(!rrpcq.empty()){
   champsim::address trigger_pc = rrpcq.front();
   rrpcq.pop();


   target_table.fill({trigger_pc, current_pc}); //target pc is the current pc
 }
 //updating Target Table for targets


 //1 treat current PC as trigger as research paper says to look in TT
 auto target_entry = target_table.check_hit({ip, {}}); //"PC of the acces is treated as a trigger PC"
 if(!target_entry.has_value()){
   target_entry = {ip,ip};
 }
 if(target_entry.has_value()){
   // If hit in target table, get the target PC
   champsim::address target_pc = target_entry->target_pc;
   champsim::address trigger_pc = target_entry->trigger_pc;  // This is the actual trigger PC


   //2 search APT with the target PC
   auto addr_pred = addr_pred_table.check_hit({target_pc, {}, 0, PREFETCH_DEGREE});


   if(addr_pred.has_value()){
     champsim::block_number last_addr = addr_pred->last_cl_addr;
     champsim::block_number::difference_type stride = addr_pred->last_stride;


     if (stride >= 0) {
       for (int i = 1; i <= addr_pred->degree; ++i) {
         champsim::address pf_addr{last_addr + (stride * i)};
        
         //Issue prefetch
         bool prefetch_queued = prefetch_line(pf_addr, true, 0); //if capacity is there
        
         if (prefetch_queued && champsim::page_number{pf_addr} == champsim::page_number{addr}) { //and PF ADDRESS PAGE IS ON THE SAME PAGE IS ADDR
           //3 Record in IPT with trigger PC from target table
           ipt_table.fill({trigger_pc, pf_addr});
           active_lookahead = lookahead_entry{pf_addr, stride, PREFETCH_DEGREE, intern_->current_cycle()};
         }
       }
     }
   }
 }


 //Update address prediction table
 champsim::block_number current_cl_addr{addr};
 auto current_entry = addr_pred_table.check_hit({ip, {}, 0, PREFETCH_DEGREE});
 champsim::block_number::difference_type new_stride = 0;
 if (current_entry.has_value()) {
     new_stride = champsim::offset(current_entry->last_cl_addr, current_cl_addr);
 }
 addr_pred_table.fill({ip, current_cl_addr, new_stride, PREFETCH_DEGREE});
            
 return metadata_in;
}


//issues prefetches -- based on previously decided prefetches
void t_skid::prefetcher_cycle_operate()
{
 auto current_cycle = intern_->current_cycle();


 //Check if there's an active lookahead entry ready to be issued
 if (active_lookahead.has_value() && active_lookahead->issue_cycle <= current_cycle) {
   auto [pf_address, stride, degree, issue_cycle] = active_lookahead.value();


   //Issue the prefetch
   bool success = prefetch_line(pf_address, true, 0);
  
   if (success) {
     //Prefetch issued, reset the active_lookahead
     active_lookahead.reset();
   }
 }
}
//this checks the page number of the address im going to prefetch. it makes sure wherever Im prefetching is on the page as the access that issued the prefetch


//load fetched data after a miss into cache
uint32_t t_skid::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in)
{
 //RRPCQ record PCs that triggered a prefetch recently filled into the cache.
 if(prefetch){
   //create an ipt_entry with the addr (prefetch address) to check for a hit
   ipt_entry search_entry{.pf_addr = addr}; //C++20 feature ***
  
   // Check if there's an entry in IPT for this address
   auto found = ipt_table.check_hit(search_entry);


   if(found.has_value()){
     //if we find the trigger PC associated with the prefetch in IPT...
     champsim::address trigger_pc = found->trigger_pc;


     if (rrpcq.size() >= RRPCQ_SIZE) {
       rrpcq.pop();  // Remove the oldest entry if full
     }


     rrpcq.push(trigger_pc); //add the new trigger PC


     //feed target table with the trigger PC
     if(!rrpcq.empty()){
       champsim::address target_pc = rrpcq.front();
       target_table.fill({trigger_pc, target_pc});
     }
   }
 }
 return metadata_in;
}
