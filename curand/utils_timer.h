#ifndef NULL
#def NULL ((void *)0)
#endif
 
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <ctime>
#  include <sys/time.h>
#endif
 
namespace utils {
 
class stopclock
{
public:
    stopclock() {}
    virtual ~stopclock() {;}
 
public:
    virtual void startTimer() = 0;
    virtual void stopTimer() = 0;
    virtual void resetTimer() = 0;
    virtual const double& getTimeInMilliseconds() = 0;
};
 
#ifdef _WIN32
class winStopClock : public stopclock
{
public:
 
    winStopClock() {
        unsigned __int64 freq;
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        timerFrequency = (1.0/freq);
        active = false;
        endTime = startTime = 0;
    }
    ~winStopClock() {;}
 
    inline void startTimer() { QueryPerformanceCounter((LARGE_INTEGER *)&startTime);
                               endTime = 0;
                               active = true; }
    inline void stopTimer() {
                                if (active)
                                {
                                    QueryPerformanceCounter((LARGE_INTEGER *)&endTime);
                                    timeDifferenceInMilliseconds = ((endTime-startTime) * timerFrequency);
                                    active = false;
                                }
                            }
    inline void resetTimer() { QueryPerformanceCounter((LARGE_INTEGER *) &startTime);
                               endTime = 0;
                               active = true; }
    inline const double& getTimeInMilliseconds() { if (active) stopTimer(); active = true;
                                                   return timeDifferenceInMilliseconds; }
 
private:
    unsigned __int64 startTime, endTime;
    double timerFrequency, timeDifferenceInMilliseconds;
    bool active;
};
 
#else
 
class linuxStopClock : public stopclock
{
public:
 
    linuxStopClock() {
        active = false;
        endTime = startTime = 0;
    }
    ~linuxStopClock() {;}
 
    inline void startTimer() { gettimeofday(&tim, NULL);
                               startTime=tim.tv_sec+(tim.tv_usec * 0.0000001);
                               endTime = 0;
                               active = true; }
    inline void stopTimer() {
                                if (active)
                                {
                                    gettimeofday(&tim, NULL);
                                    endTime=tim.tv_sec+(tim.tv_usec * 0.0000001);
                                    active = false;
                                }
                            }
    inline void resetTimer() { gettimeofday(&tim, NULL);
                               startTime=tim.tv_sec+(tim.tv_usec * 0.0000001);
                               endTime = 0;
                               active = true; }
    inline const double& getTimeInMilliseconds() { if (active) stopTimer(); active = true;
                                                   tref = endTime-startTime;
                                                   return tref; }
 
private:
    struct timeval tim;
    double startTime, endTime ,tref;
    bool active;
};
 
#endif
 
inline bool createTimer(stopclock **_timer)
{
#ifdef _WIN32
    *_timer = (stopclock *)new winStopClock();
#else
    *_timer = (stopclock *)new linuxStopClock();
#endif
    return (*_timer != NULL) ? true : false;
}
 
inline bool removeTimer(stopclock **_timer)
{
    if (*_timer)
    {
        delete *_timer;
        *_timer = NULL;
    }
    return true;
}
}
