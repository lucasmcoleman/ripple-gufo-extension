// Stage profile accumulator and report (hip/profile.*) with a fake clock: no
// sleeping, no device. The executor wiring is checked by the host-ops run in
// docs/PERF-PROFILE.md.
#include <cstdint>
#include <iostream>
#include <string>

#include "hip/profile.hpp"
#include "tests/check.hpp"

using namespace gufo::models::ricursive::profile;
using ricursive_test::Require;
using ricursive_test::RequireContains;

namespace {

struct FakeClock {
  std::int64_t now{1000};
  ClockFn Fn() {
    return [this] { return now; };
  }
};

constexpr std::int64_t kMs = 1'000'000;

/// Runs of blanks become one, so the checks do not depend on column widths.
std::string Squeeze(const std::string& text) {
  std::string out;
  for (const char ch : text) {
    if (ch == ' ' && !out.empty() && out.back() == ' ') continue;
    out += ch;
  }
  return out;
}

void TestBuckets() {
  Require(BucketFor(2, 0) == Bucket::kPrompt, "2 rows is a prompt pass");
  Require(BucketFor(4096, 99999) == Bucket::kPrompt, "many rows is a prompt pass");
  Require(BucketFor(1, 1) == Bucket::kGenLt1024, "ctx 1");
  Require(BucketFor(1, 1023) == Bucket::kGenLt1024, "ctx 1023");
  Require(BucketFor(1, 1024) == Bucket::kGen1024, "ctx 1024");
  Require(BucketFor(1, 2047) == Bucket::kGen1024, "ctx 2047");
  Require(BucketFor(1, 2048) == Bucket::kGen2048, "ctx 2048");
  Require(BucketFor(1, 4095) == Bucket::kGen2048, "ctx 4095");
  Require(BucketFor(1, 4096) == Bucket::kGen4096, "ctx 4096");
  Require(BucketFor(1, 100000) == Bucket::kGen4096, "ctx 100000");
  Require(IsMatmul(Id(true, LayerStage::kProjQ)) && IsMatmul(Id(false, LayerStage::kProjFfnDown)) &&
              IsMatmul(Id(PassStage::kOutputProj)),
          "projections are matmul stages");
  Require(!IsMatmul(Id(true, LayerStage::kAttention)) && !IsMatmul(Id(PassStage::kEmbed)),
          "attention and embedding are not");
  Require(StageName(Id(true, LayerStage::kAttention)) == "sw.attention", "sliding name");
  Require(StageName(Id(false, LayerStage::kAttention)) == "gl.attention", "global name");
  Require(StageName(Id(PassStage::kLogitsReadback)) == "logits_readback", "pass name");
}

void TestAccumulation() {
  FakeClock clk;
  StageProfile p(clk.Fn());
  Require(p.Report().empty() && p.TotalPasses() == 0, "empty profile has an empty report");

  // Two 1-row passes at ctx 100: attention 3 ms + 5 ms, output projection 10 ms
  // (2000 MB read) each, 1 ms unmarked each.
  for (int i = 0; i < 2; ++i) {
    p.BeginPass(1, 100 + static_cast<std::uint32_t>(i));
    clk.now += 3 * kMs;
    p.Mark(Id(true, LayerStage::kAttention));
    clk.now += 5 * kMs;
    p.Mark(Id(false, LayerStage::kAttention));
    clk.now += 10 * kMs;
    p.Mark(Id(PassStage::kOutputProj), 2'000'000'000ULL);
    clk.now += 1 * kMs;  // never marked: shows up as unattributed
    p.EndPass();
  }
  // An aborted pass (no EndPass) leaves no trace; the next BeginPass drops it.
  p.BeginPass(1, 100);
  clk.now += 777 * kMs;
  p.Mark(Id(true, LayerStage::kAttnNorm));
  // A prompt pass of 8 rows.
  p.BeginPass(8, 8);
  clk.now += 20 * kMs;
  p.Mark(Id(PassStage::kEmbed));
  p.EndPass();
  // Generation passes in the other context buckets.
  p.BeginPass(1, 1500);
  clk.now += 4 * kMs;
  p.Mark(Id(true, LayerStage::kAttention));
  p.EndPass();
  p.BeginPass(1, 5000);
  clk.now += 6 * kMs;
  p.Mark(Id(true, LayerStage::kAttention));
  p.EndPass();

  Require(p.TotalPasses() == 5, "5 completed passes");
  const BucketStats& g = p.Stats(Bucket::kGenLt1024);
  Require(g.passes == 2 && g.rows == 2, "gen<1024 counts");
  Require(g.ns[Id(true, LayerStage::kAttention)] == 6 * kMs, "sliding attention total");
  Require(g.ns[Id(false, LayerStage::kAttention)] == 10 * kMs, "global attention total");
  Require(g.ns[Id(true, LayerStage::kAttnNorm)] == 0, "aborted pass not counted");
  Require(g.total_ns == 2 * 19 * kMs, "pass totals include unmarked time");
  Require(g.syncs == 6, "3 marks per pass");
  Require(g.bytes[Id(PassStage::kOutputProj)] == 4'000'000'000ULL, "weight bytes add up");
  Require(p.Stats(Bucket::kPrompt).passes == 1 && p.Stats(Bucket::kPrompt).rows == 8, "prompt bucket");
  Require(p.Stats(Bucket::kGen1024).passes == 1, "gen 1024-2047 bucket");
  Require(p.Stats(Bucket::kGen2048).passes == 0, "gen 2048-4095 empty");
  Require(p.Stats(Bucket::kGen4096).passes == 1, "gen >= 4096 bucket");

  const std::string full = p.Report();
  std::cout << full;
  const std::string r = Squeeze(full);
  RequireContains(r, "# bucket gen_lt1024: 2 passes, 2 rows", "bucket header");
  RequireContains(r, "# bucket prompt: 1 passes, 8 rows", "prompt header");
  RequireContains(r, "# bucket gen_ge4096: 1 passes", "ge4096 header");
  Require(full.find("gen_2048_4095") == std::string::npos, "empty buckets are not printed");
  // 2 passes x 19 ms: attention sw 6 ms total / 3.0 ms mean / 15.79 % of the pass.
  RequireContains(r, "gen_lt1024 sw.attention 6.000 3.0000 15.79", "sw.attention line");
  RequireContains(r, "gen_lt1024 gl.attention 10.000 5.0000 26.32", "gl.attention line");
  // 2000 MB per pass in 10 ms = 200 GB/s.
  RequireContains(r, "gen_lt1024 output_proj 20.000 10.0000 52.63 2000.00 200.0", "bytes and GB/s");
  RequireContains(r, "gen_lt1024 unattributed 2.000 1.0000 5.26", "unattributed line");
  RequireContains(r, "gen_lt1024 mean_ms_per_pass 19.000", "mean ms");
  // 2 rows in 38 ms.
  RequireContains(r, "gen_lt1024 tokens_per_s 52.632", "tokens per second");
  RequireContains(r, "gen_lt1024 sync_points_per_pass 3.0", "sync points");
}

}  // namespace

int main() {
  return ricursive_test::Run([] {
    TestBuckets();
    TestAccumulation();
    std::cout << "profile: buckets, accumulation, aborted pass, report columns\n";
  });
}
