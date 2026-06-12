# ML-infra platform tasks. Run from the repo root: `make -f deploy/Makefile <target>`.
IMAGE ?= discovery-agents-ml:latest
WEB_IMAGE ?= discovery-agents-web:latest
PORT ?= 8000
ARTIFACTS ?= outputs/ml

.PHONY: install curate train io-bench benchmark export smoke docker-build docker-run k8s-apply \
        web-install web-run web-docker-build web-docker-run

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

# --- FastAPI live demo ---
web-install:
	pip install -e ".[web,gemini]"

# Local dev server with autoreload. Set GEMINI_API_KEY in the env for free server runs.
web-run:
	uvicorn discovery_agents.webapp:app --reload --port $(PORT)

web-docker-build:
	docker build -f deploy/Dockerfile.web -t $(WEB_IMAGE) .

web-docker-run:
	docker run --rm -p $(PORT):8000 -e GEMINI_API_KEY=$${GEMINI_API_KEY} $(WEB_IMAGE)
