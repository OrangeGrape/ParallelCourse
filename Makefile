FLAGS=-I/usr/include -L/usr/lib/x86_64-linux-gnu
LIB=-lOpenCL
all: Histogram.x

%.x: %.c
	cc ${FLAGS} -o $@ $< ${LIB}

clean:
	rm -f *.x

