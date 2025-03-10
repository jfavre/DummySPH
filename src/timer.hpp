// https://github.com/sphexa-org/sphexa.git
#pragma once

#include <chrono>
#include <iostream>

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#if defined(USE_PROFILING_NVTX) || defined(USE_PROFILING_SCOREP)

#ifdef USE_PROFILING_NVTX
#include <nvToolsExt.h>
#define MARK_BEGIN(xx) nvtxRangePush(xx);
#define MARK_END nvtxRangePop();
#endif

#ifdef USE_PROFILING_SCOREP
#include "scorep/SCOREP_User.h"
#define MARK_BEGIN(xx)                                                                                                 \
    {                                                                                                                  \
        SCOREP_USER_REGION(xx, SCOREP_USER_REGION_TYPE_COMMON)
#define MARK_END }
#endif

#else
#define MARK_BEGIN(xx)
#define MARK_END
#endif

namespace sphexa
{

class Timer
{
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<float>       Time;

public:
    Timer(std::ostream& out)
        : out(out)
    {
    }

    void start()
    {
        numStartCalled++;
        tstart = tlast = Clock::now();
    }

    void step(const std::string& name)
    {
        auto now = Clock::now();
        stepTimes.push_back(stepDuration(now));
        if (!name.empty()) {
            out << "# " << name << ": " << stepTimes.back() << "s "; // << std::endl;
        }
        tlast = now;
        //
        auto memInfo = read_proc_meminfo({"MemTotal", "MemFree", "MemAvailable"}, 3);
        if (!name.empty()) {
            out << "/ MemFree = " << memInfo["MemFree"] << " / MemAvailable = " << memInfo["MemAvailable"] << " / MemTotal = " << memInfo["MemTotal"] << " kB" << std::endl;
        }
    }

    //! @brief time elapsed between tstart and last call of step()
    [[nodiscard]] float sumOfSteps() const { return std::chrono::duration_cast<Time>(tlast - tstart).count(); }

    //! @brief time elapsed between tstart and now
    [[nodiscard]] float elapsed() const { return std::chrono::duration_cast<Time>(Clock::now() - tstart).count(); }

    std::unordered_map<std::string, long> read_proc_meminfo(const std::unordered_set<std::string>& keys, size_t max_lines = 3) {
        std::ifstream meminfo("/proc/meminfo", std::ios::in);
        if (!meminfo) {
            throw std::runtime_error("Failed to open /proc/meminfo");
        }
    
        std::unordered_map<std::string, long> memData;
        std::string line;
        size_t lines_read = 0;
    
        while (lines_read < max_lines && std::getline(meminfo, line)) {
            std::istringstream iss(line);
            std::string key;
            long value;
            std::string unit;
    
            if (iss >> key >> value >> unit) {
                key.pop_back();  // Remove trailing ':'
                if (keys.find(key) != keys.end()) {
                //c++20: if (keys.contains(key)) {
                    memData[key] = value;
                    if (memData.size() == keys.size()) break;
                    // Stop early if all needed keys are found
                }
            }
            lines_read++;
        }
    
        return memData;
    }

    /*
    template<class Archive>
    void writeTimings(Archive* ar, const std::string& outFile)
    {
        ar->addStep(0, stepTimes.size(), outFile + ar->suffix());
        int numRanks = ar->numRanks();
        ar->stepAttribute("numRanks", &numRanks, 1);
        ar->stepAttribute("numIterations", &numStartCalled, 1);
        ar->writeField("timings", stepTimes.data(), stepTimes.size());
        ar->closeStep();

        numStartCalled = 0;
        stepTimes.clear();
    }
    */

private:
    //c++17 or compile with -fconcepts
    template <typename T>
    float stepDuration(T now) {
        return std::chrono::duration_cast<Time>(now - tlast).count();
    }
    //c++20 float stepDuration(auto now) { return std::chrono::duration_cast<Time>(now - tlast).count(); }
    // std::unordered_map<std::string, long> procMemInfo = read_proc_meminfo({"MemTotal", "MemFree", "MemAvailable"}, 3);

    std::ostream&                  out;
    std::chrono::time_point<Clock> tstart, tlast;
    std::vector<float>             stepTimes;
    int                            numStartCalled{0};
};

} // namespace sphexa
