/* SPDX-License-Identifier: LGPL-2.1-or-later
 * Reference harness adapted from atracdenc on 2026-11-19.
 * Test tool only; never used by the Python encoder at runtime.
 * See NOTICE.md for upstream authors and revision.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <vector>
#include "compressed_io.h"
#include "atrac/atrac3plus_pqf/atrac3plus_pqf.h"
#include "atrac/at3p/at3p_mdct.h"
#include "atrac/at3p/at3p_bitstream.h"
#include "atrac/at3p/at3p_tables.h"
#include "atrac/atrac_scale.h"
using namespace NAtracDEnc;
class Output : public ICompressedOutput {
public:
 void WriteFrame(std::vector<char> data) override { fwrite(data.data(),1,data.size(),stdout); }
 std::string GetName() const override { return {}; }
 size_t GetChannelNum() const override { return 2; }
};
int main(int argc,char**argv){
 if(argc<2)return 2;
 const bool packets=std::string(argv[1])=="packets";
 int channels=packets ? 2 : 1;
 Output out;
 TAt3PBitStream writer(&out,argc>2?std::atoi(argv[2]):744);
 TScaler<NAt3p::TScaleTable> scaler;
 std::vector<at3plus_pqf_a_ctx_t> pqf;
 std::vector<TAt3pMDCT::THistBuf> history(channels);
 std::vector<std::array<float,2048>> previous(channels),pending(channels);
 for(int ch=0;ch<channels;ch++)pqf.push_back(at3plus_pqf_create_a_ctx());
 TAt3pMDCT mdct;
 std::vector<float> pcm(channels*2048),bands(2048),scaled(2048),spectra(2048);
 bool started=false;
 while(fread(pcm.data(),sizeof(float),pcm.size(),stdin)==pcm.size()){
  std::vector<TAt3PBitStream::TSingleChannelElement> elements(channels);
  for(int ch=0;ch<channels;ch++){
   at3plus_pqf_do_analyse(pqf[ch],pcm.data()+ch*2048,bands.data());
   if(!packets)fwrite(bands.data(),sizeof(float),2048,stdout);
   const float* x=packets?pending[ch].data():bands.data();
   for(int i=0;i<2048;i++)scaled[i]=x[i]/(32768.0/1.122018);
   if(!packets||started){
    TAt3pMDCT::TPcmBandsData ptrs;
    for(int b=0;b<16;b++)ptrs[b]=scaled.data()+b*128;
    mdct.Do(spectra.data(),ptrs,history[ch],{});
    if(!packets)fwrite(spectra.data(),sizeof(float),2048,stdout);
    else elements[ch].ScaledBlocks=scaler.ScaleFrame(spectra,NAt3p::TScaleTable::TBlockSizeMod());
   }
   if(started)pending[ch]=previous[ch];
   memcpy(previous[ch].data(),bands.data(),2048*sizeof(float));
  }
  if(packets&&started)writer.WriteFrame(channels,nullptr,elements);
  started=true;
 }
 for(auto p:pqf)at3plus_pqf_free_a_ctx(p);
}
