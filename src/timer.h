/**************************************************************
This code is a part of a course on cuda taught by the author: 
Lokman A. Abbas-Turki

Those who re-use this code should mention in their code 
the name of the author above.
***************************************************************/

#ifndef SEEK_SET
#define SEEK_SET 0
#endif
#ifndef CLOCKS_PER_SEC
#include <unistd.h>
#define CLOCKS_PER_SEC _SC_CLK_TCK
#endif

class Timer { 

	private:
	double _start;
    double t;

	public:
    Timer(void) : _start(0.0), t(0.0) {
	}


	public:
	inline void start(void) {
		_start = clock();
	}

	inline void add(void) {
		t = (clock() - _start)/(double)CLOCKS_PER_SEC;
    }

	inline double getstart(void) const {
		return _start;
	}

    inline double getsum(void) const {
		return t;
	}

    inline void reset(void) {
		t = 0.0;
	}

};