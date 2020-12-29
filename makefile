#defination for compiler 
CC= nvcc -std=c++14 -arch=compute_61 #-g -G

ifndef KONG_BUILD
KONG_BUILD = false
endif

ifeq ($(KONG_BUILD), true)
ifndef ARCH
$(error ARCH is not set; has to be set to either Pascal or Titan)
endif
endif

#defination for compiler flag
CFLAG:= -Xcompiler -static-libgcc -Xcompiler -static-libstdc++ -Xcompiler -fopenmp -x cu -Xcudafe "--diag_suppress=useless_type_qualifier_on_return_type" -Xcudafe "--diag_suppress=code_is_unreachable" -Xcudafe "--diag_suppress=set_but_not_used" -Xcudafe "--diag_suppress=integer_sign_change" -dc

SRCDIR:= src
DEMODIR:= demo
OBJDIR := build
MATHDIR:= math
LATTICEDIR:= lattice
TRFMDIR:= transfrm
PKEDIR:= pke

ifeq ($(KONG_BUILD), true)
	PALISADEDIR:= /usr/local
	export CPLUS_INCLUDE_PATH:=$(PALISADEDIR)/include/palisade
	PALINCLUDES:= $(shell find $(PALISADEDIR)/include/palisade -name "pke" -o -name "core" -o -name "cereal" | sed -e 's/^/-I /'| xargs)
	LINKDIR:= -L$(PALISADEDIR)/lib
	ifeq ($(ARCH), Pascal)
		CC= nvcc -std=c++14 -arch=compute_60
	endif
	ifeq ($(ARCH), Titan)
		CC= nvcc -std=c++14 -arch=compute_75
	endif	
else
	PALISADEDIR:= /usr/local/palisade
	export CPLUS_INCLUDE_PATH:=$(PALISADEDIR)/include
	PALINCLUDES:= $(shell find $(PALISADEDIR)/include -name "pke" -o -name "core" -o -name "cereal" | sed -e 's/^/-I /'| xargs)
	LINKDIR:= -L$(PALISADEDIR)/lib
endif

$(info KONG_BUILD is set to "$(KONG_BUILD)")
ifeq ($(KONG_BUILD), true)
$(info ARCHITECTURE is set to "$(ARCH)")
endif	

SRCOBJS := $(patsubst %.cpp, $(OBJDIR)/%.o, $(wildcard $(SRCDIR)/*.cpp))
MATHOBJS := $(patsubst %.cpp, $(OBJDIR)/%.o, $(wildcard $(SRCDIR)/$(MATHDIR)/*.cpp))
LATTICEOBJS := $(patsubst %.cpp, $(OBJDIR)/%.o, $(wildcard $(SRCDIR)/$(LATTICEDIR)/*.cpp))
PKEOBJS:= $(patsubst %.cpp, $(OBJDIR)/%.o, $(wildcard $(SRCDIR)/$(PKEDIR)/*.cpp))
TRANSFRMOBJS := $(patsubst %.cpp, $(OBJDIR)/%.o, $(wildcard $(SRCDIR)/$(TRFMDIR)/*.cpp))
DEMOOBJS := $(patsubst %.cpp, $(OBJDIR)/%.o, $(wildcard $(DEMODIR)/*.cpp))
EXES:= $(patsubst %.o, %, $(DEMOOBJS))
INCLUDES:= -I$(SRCDIR)/$(MATHDIR) -I$(SRCDIR)/$(LATTICEDIR) -I$(SRCDIR)/$(TRFMDIR) -I$(SRCDIR)/$(PKEDIR) -I$(SRCDIR) 
INCLUDES+= $(PALINCLUDES)
ALLOBJS:= $(MATHOBJS) $(LATTICEOBJS) $(TRANSFRMOBJS) $(PKEOBJS) $(SRCOBJS)

#build for math object files
$(OBJDIR)/$(SRCDIR)/$(MATHDIR)/%.o : $(SRCDIR)/$(MATHDIR)/%.cpp $(SRCDIR)/$(MATHDIR)/%.h | $(OBJDIR)/$(SRCDIR)/$(MATHDIR)
	$(CC) $(CFLAG) -I$(INCLUDES)/ -c $< -o $@ 

#build for lattice object files
$(OBJDIR)/$(SRCDIR)/$(LATTICEDIR)/%.o : $(SRCDIR)/$(LATTICEDIR)/%.cpp $(SRCDIR)/$(LATTICEDIR)/%.h | $(OBJDIR)/$(SRCDIR)/$(LATTICEDIR)
	$(CC) $(CFLAG) -I$(INCLUDES)/ -c $< -o $@ 

#build for pke object files	
$(OBJDIR)/$(SRCDIR)/$(PKEDIR)/%.o : $(SRCDIR)/$(PKEDIR)/%.cpp $(SRCDIR)/$(PKEDIR)/%.h | $(OBJDIR)/$(SRCDIR)/$(PKEDIR)
	$(CC) $(CFLAG) -I$(INCLUDES)/ -c $< -o $@ 
	
#build for transfrm object files
$(OBJDIR)/$(SRCDIR)/$(TRFMDIR)/%.o : $(SRCDIR)/$(TRFMDIR)/%.cpp $(SRCDIR)/$(TRFMDIR)/%.h | $(OBJDIR)/$(SRCDIR)/$(TRFMDIR)
	$(CC) $(CFLAG) -I$(INCLUDES)/ -c $< -o $@ 	
	
#build for src object files	
$(OBJDIR)/$(SRCDIR)/%.o : $(SRCDIR)/%.cpp | $(OBJDIR)/$(SRCDIR)
	$(CC) $(CFLAG) $(INCLUDES) -c $< -o $@ 	
	
#build for demo object files
$(OBJDIR)/$(DEMODIR)/%.o : $(DEMODIR)/%.cpp | $(OBJDIR)/$(DEMODIR)
	$(CC) $(CFLAG) $(INCLUDES) -c $< -o $@ 	

#Link the demo with libraries
$(OBJDIR)/$(DEMODIR)/%: $(OBJDIR)/$(DEMODIR)/%.o
	$(CC) $(ALLOBJS) $^ --cudart=static -o $@ $(LINKDIR) -lcurand_static -lculibos -lPALISADEcore_static -lgomp /usr/lib/gcc/x86_64-linux-gnu/7.5.0/libquadmath.a

all : $(MATHOBJS) $(LATTICEOBJS) $(TRANSFRMOBJS) $(PKEOBJS) $(SRCOBJS) $(DEMOOBJS) $(EXES)

	
$(OBJDIR)/$(SRCDIR): 
	mkdir -p $(OBJDIR)/$(SRCDIR)
	
$(OBJDIR)/$(DEMODIR): 
	mkdir -p $(OBJDIR)/$(DEMODIR)

$(OBJDIR)/$(SRCDIR)/$(MATHDIR): 
	mkdir -p $(OBJDIR)/$(SRCDIR)/$(MATHDIR)	

$(OBJDIR)/$(SRCDIR)/$(LATTICEDIR): 
	mkdir -p $(OBJDIR)/$(SRCDIR)/$(LATTICEDIR)	
	
$(OBJDIR)/$(SRCDIR)/$(PKEDIR): 
	mkdir -p $(OBJDIR)/$(SRCDIR)/$(PKEDIR)			
	
$(OBJDIR)/$(SRCDIR)/$(TRFMDIR): 
	mkdir -p $(OBJDIR)/$(SRCDIR)/$(TRFMDIR)			
	
.PHONY: 
	clean	
	
clean:
	rm -rf $(OBJDIR)	
