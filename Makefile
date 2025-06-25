TESTS = $(basename $(notdir $(wildcard tests/test_*.py)))

BENCHMARKS = $(basename $(notdir $(wildcard benchmarks/benchmark_*.py)))
BENCHMARK_ARGS =
CUDAGRAPH = 0
PROFILE = 0



.PHONY: tests
tests:
	@echo "Running all tests"
	python -m pytest

$(TESTS):
	@echo "Running $@"
	python -m pytest -k $@


.PHONY: benchmarks
benchmarks: $(BENCHMARKS)

ifeq ($(PROFILE),1)
BENCHMARK_MSG = Profiling
BENCHMARK_ARGS += --profile
else
BENCHMARK_MSG = Running
endif

ifeq ($(CUDAGRAPH),1)
BENCHMARK_CUDAGRAPH_MSG = with cudagraph
BENCHMARK_ARGS += --cudagraph
endif

$(BENCHMARKS):
	@echo "$(BENCHMARK_MSG) $@ $(BENCHMARK_CUDAGRAPH_MSG)"
	python benchmarks/$@.py $(BENCHMARK_ARGS)


.PHONY: clean
clean:
	rm -rf .pytest_cache
