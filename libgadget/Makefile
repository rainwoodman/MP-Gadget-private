# vim: set ft=make:

# Customization; see Options.mk.example

CONFIG ?= ../Options.mk

include $(CONFIG)

UTILS_TESTED = memory openmpsort interp peano
UTILS_MPI_TESTED = mpsort

TESTED = hci \
	slotsmanager \
	powerspectrum \
	cosmology \
	cooling \
	forcetree \
	timefac \
	timebinmgr \
	neutrinos_lra \
	omega_nu_single \
	metal_return \
	cooling_rates \
	density \
	gravity \
	exchange

MPI_TESTED = exchange fof

TESTBIN :=$(UTILS_TESTED:%=.objs/utils/test_%) $(UTILS_MPI_TESTED:%=.objs/utils/test_%) $(TESTED:%=.objs/test_%) $(MPI_TESTED:%=.objs/test_%)
MPISUITE = $(MPI_TESTED:%=test_%) $(UTILS_MPI_TESTED:%=utils/test_%)
SUITE = $(TESTED:%=test_%) $(UTILS_TESTED:%=utils/test_%) $(MPISUITE)
include ../Makefile.rules

FILES = $(shell git ls-files)

# define the version
# (easier to extract from other utils, e.g. python)
include ../Makefile.version

# Main Rules
#
# make V=1 to see the command line
#

GADGET_OBJS =  \
	 gdbtools.o hci.o\
	 fof.o fofpetaio.o petaio.o \
	 domain.o exchange.o slotsmanager.o partmanager.o \
	 blackhole.o bhinfo.o bhdynfric.o \
	 timebinmgr.o \
	 run.o drift.o stats.o \
	 timestep.o init.o checkpoint.o \
	 sfr_eff.o cooling.o cooling_rates.o cooling_uvfluc.o cooling_qso_lightup.o \
	 winds.o veldisp.o density.o metal_return.o \
	 treewalk.o cosmology.o \
	 gravshort-tree.o gravshort-pair.o hydra.o  timefac.o \
	 gravpm.o powerspectrum.o \
	 forcetree.o sidm.o \
	 petapm.o gravity.o \
	 densitykernel.o lightcone.o walltime.o\
	 runtests.o \
	 neutrinos_lra.o \
     omega_nu_single.o \
	 config.o \
	 uvbg.o \
	 plane.o\
	 lenstools.o

GADGET_UTILS_OBJS= \
utils/endrun.o \
utils/peano.o \
utils/interp.o \
utils/memory.o \
utils/mpsort.o \
utils/mymalloc.o \
utils/system.o \
utils/paramset.o \
utils/event.o \
utils/openmpsort.o \
utils/unitsystem.o \
utils/string.o \
utils/spinlocks.o

GADGET_OBJS := $(GADGET_OBJS:%=.objs/%)
GADGET_UTILS_OBJS := $(GADGET_UTILS_OBJS:%=.objs/%)

all: libgadget.a libgadget-utils.a

.PHONY: all test run-tests

.objs/utils/test_%: tests/test_%.c .objs/utils/%.o ../tests/stub.c ../tests/cmocka.c libgadget-utils.a
	$(MPICC) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_%: tests/test_%.c .objs/%.o ../tests/stub.c ../tests/cmocka.c libgadget-utils.a
	$(MPICC) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_timebinmgr: tests/test_timebinmgr.c .objs/timebinmgr.o .objs/cosmology.o .objs/omega_nu_single.o ../tests/stub.c ../tests/cmocka.c libgadget-utils.a
	$(MPICC) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_neutrinos_lra: tests/test_neutrinos_lra.c .objs/neutrinos_lra.o .objs/cosmology.o .objs/omega_nu_single.o ../tests/stub.c ../tests/cmocka.c libgadget-utils.a

.objs/test_cooling_rates: tests/test_cooling_rates.c .objs/cooling_rates.o .objs/cooling_uvfluc.o ../tests/stub.c ../tests/cmocka.c libgadget-utils.a
	$(MPICC) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_exchange: tests/test_exchange.c .objs/exchange.o ../tests/stub.c ../tests/cmocka.c libgadget.a libgadget-utils.a
	$(MPICC) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_density: tests/test_density.c .objs/density.o libgadget.a ../tests/stub.c ../tests/cmocka.c libgadget-utils.a
	$(MPICC) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_metal_return: tests/test_metal_return.c .objs/metal_return.o libgadget.a ../tests/stub.c ../tests/cmocka.c libgadget-utils.a
	$(MPICC) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_cooling: tests/test_cooling.c .objs/cooling.o .objs/cooling_rates.o .objs/cooling_uvfluc.o ../tests/stub.c ../tests/cmocka.c libgadget-utils.a
	$(MPICC) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_gravity: tests/test_gravity.c libgadget.a ../tests/stub.c ../tests/cmocka.c libgadget-utils.a
	$(MPICC) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_fof: tests/test_fof.c libgadget.a ../tests/stub.c ../tests/cmocka.c libgadget-utils.a
	$(MPICC) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

.objs/test_forcetree: tests/test_forcetree.c libgadget.a ../tests/stub.c ../tests/cmocka.c libgadget-utils.a
	$(MPICC) $(TCFLAGS) -I../tests/ $^ $(LIBS) -o $@

build-tests: $(TESTBIN)

test : build-tests
	trap 'err=1' ERR; for tt in $(SUITE) ; do \
		if [[ "$(MPISUITE)" =~ .*$$tt.* ]]; then \
            OMPI_MCA_rmaps_base_oversubscribe=1 \
			mpirun -np 4 .objs/$$tt ;  \
		else \
			.objs/$$tt ; \
		fi ;  \
	done; exit $$err

libgadget-utils.a: $(GADGET_UTILS_OBJS)
	$(AR) r $@ $(GADGET_UTILS_OBJS)

libgadget.a: $(GADGET_OBJS)
	$(AR) r $@ $(GADGET_OBJS)

clean:
	rm -rf .objs libgadget.a libgadget-utils.a config.c

config.c: $(CONFIG)
	mkdir -p `dirname $@`
	MPICC="$(MPICC)" CFLAGS="$(CFLAGS)" OPT="$(OPT)" OPTIMIZE="$(OPTIMIZE)" VERSION="$(VERSION)" bash makeconfig.sh $@

GADGET_DEPS := $(GADGET_OBJS:.o=.d)
GADGET_UTILS_DEPS := $(GADGET_UTILS_OBJS:.o=.d)
-include $(GADGET_DEPS)
-include $(GADGET_UTILS_DEPS)


#This snippet works out the current git revision and the git revision in config.h.
#It checks whether they are the same.
#If they are not the same it makes config.h as a PHONY target to be rebuilt.
GIT=$(subst -,_,$(shell git describe --always --dirty --abbrev=10))
EXISTGIT=$(shell grep "GADGET_VERSION" config.c)
ifeq (,$(findstring $(GIT),$(EXISTGIT)))
.PHONY: config.c
endif

MPI_SHOWME := $(shell $(MPICC) -showme)

.objs/%.compdb_entry: %.c
	@echo "    {" > $@
	@echo "        \"command\": \"$(MPI_SHOWME) $(CFLAGS) -c $<\","   >> $@
	@echo "        \"directory\": \"$(CURDIR)\","               >> $@
	@echo "        \"file\": \"$<\""                    >> $@
	@echo "    },"                              >> $@

COMPDB_ENTRIES = $(addsuffix .compdb_entry, $(basename $(GADGET_OBJS)))

compile_commands.json: $(COMPDB_ENTRIES)
	@echo "[" > $@.tmp
	@cat $^ >> $@.tmp
	@sed '$$d' < $@.tmp > $@
	@echo "    }" >> $@
	@echo "]" >> $@
	@rm $@.tmp
