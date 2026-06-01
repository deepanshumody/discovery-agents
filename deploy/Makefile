# ML-infra platform tasks. Run from the repo root: `make -f deploy/Makefile <target>`.
IMAGE ?= discovery-agents-ml:latest
ARTIFACTS ?= outputs/ml

.PHONY: install curate train io-bench benchmark export smoke docker-build docker-run k8s-apply

install:
	pip install -e ".[dev,ml,dask,benchmark]"

curate:
	python -m discovery_agents.mlinfra.cli curate --artifacts $(ARTIFACTS)

train:
	python -m discovery_agents.mlinfra.cli train --artifacts $(ARTIFACTS)

io-bench:
	python -m discovery_agents.mlinfra.cli io-bench --artifacts $(ARTIFACTS)

benchmark:
	python -m discovery_agents.mlinfra.cli benchmark --full --with-st

export:
	python -m discovery_agents.mlinfra.cli export --artifacts $(ARTIFACTS)

smoke:
	python -m discovery_agents.mlinfra.cli train --smoke

docker-build:
	docker build -f deploy/Dockerfile -t $(IMAGE) .

docker-run:
	docker run --rm $(IMAGE) train --smoke

k8s-apply:
	kubectl apply -f deploy/k8s/train-job.yaml
