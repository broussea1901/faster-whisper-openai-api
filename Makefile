# Makefile for Faster Whisper v2
.PHONY: help build build-cpu build-gpu run-cpu run-gpu test clean logs stop

# Default target
.DEFAULT_GOAL := help

# Variables
DOCKER_REGISTRY ?= 
IMAGE_NAME ?= faster-whisper-v2
GPU_IMAGE = $(IMAGE_NAME):gpu-diarization
CPU_IMAGE = $(IMAGE_NAME):cpu
PORT ?= 8000
API_KEYS ?= 

help: ## Show this help message
	@echo "Faster Whisper v2 - Makefile Commands"
	@echo "===================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make build-gpu       # Build GPU image with diarization"
	@echo "  make run-gpu        # Run GPU container"
	@echo "  make test           # Run API tests"

build: build-cpu build-gpu ## Build all images

build-cpu: ## Build CPU image
	@echo "🏗️  Building CPU image..."
	docker build -f Dockerfile.cpu -t $(CPU_IMAGE) .

build-gpu: requirements-gpu.txt download_models.py ## Build GPU image with diarization
	@echo "🏗️  Building GPU image with diarization..."
	docker build -f Dockerfile.gpu -t $(GPU_IMAGE) .

build-gpu-no-cache: requirements-gpu.txt download_models.py ## Build GPU image without cache
	@echo "🏗️  Building GPU image (no cache)..."
	docker build --no-cache -f Dockerfile.gpu -t $(GPU_IMAGE) .

requirements-gpu.txt: ## Create GPU requirements file
	@echo "📝 Creating requirements-gpu.txt..."
	@cat > requirements-gpu.txt << 'EOF'
	# [Copy content from requirements-gpu artifact above]
	EOF

download_models.py: ## Create model download script
	@echo "📝 Creating download_models.py..."
	@cat > download_models.py << 'EOF'
	# [Copy content from download_models.py artifact above]
	EOF

run-cpu: ## Run CPU container
	@echo "🚀 Starting CPU container..."
	docker-compose -f docker-compose.cpu.yml up -d
	@echo "✅ CPU container started at http://localhost:$(PORT)"

run-gpu: ## Run GPU container with diarization
	@echo "🚀 Starting GPU container with diarization..."
	docker-compose -f docker-compose.gpu.yml up -d
	@echo "✅ GPU container started at http://localhost:$(PORT)"
	@echo "   Diarization: ENABLED"

run-gpu-dev: ## Run GPU container in development mode
	@echo "🚀 Starting GPU container in dev mode..."
	docker run -it --rm \
		--gpus all \
		-p $(PORT):8000 \
		-e API_KEYS="$(API_KEYS)" \
		-e ENABLE_DIARIZATION=true \
		-v $(PWD):/app \
		-v whisper-models:/home/whisper/.cache \
		-v nemo-models:/models/nemo_cache \
		$(GPU_IMAGE)

test: ## Run API tests
	@echo "🧪 Running API tests..."
	@chmod +x scripts/test-api.sh
	@./scripts/test-api.sh

test-diarization: ## Test diarization specifically
	@echo "🧪 Testing diarization..."
	@curl -s -X POST "http://localhost:$(PORT)/v1/audio/transcriptions" \
		-H "Authorization: Bearer $(API_KEYS)" \
		-F "file=@examples/test.wav" \
		-F "model=whisper-1" \
		-F "timestamp_granularities=speaker" \
		-F "response_format=json" | jq '.segments[:3]'

logs: ## Show container logs
	@docker-compose -f docker-compose.gpu.yml logs -f

logs-cpu: ## Show CPU container logs
	@docker-compose -f docker-compose.cpu.yml logs -f

stop: ## Stop all containers
	@echo "🛑 Stopping containers..."
	@docker-compose -f docker-compose.cpu.yml down 2>/dev/null || true
	@docker-compose -f docker-compose.gpu.yml down 2>/dev/null || true
	@echo "✅ All containers stopped"

clean: stop ## Clean up containers and images
	@echo "🧹 Cleaning up..."
	@docker rmi $(GPU_IMAGE) $(CPU_IMAGE) 2>/dev/null || true
	@docker volume prune -f
	@echo "✅ Cleanup complete"

push: ## Push images to registry
	@if [ -z "$(DOCKER_REGISTRY)" ]; then \
		echo "❌ DOCKER_REGISTRY not set"; \
		exit 1; \
	fi
	@echo "📤 Pushing images to $(DOCKER_REGISTRY)..."
	docker tag $(GPU_IMAGE) $(DOCKER_REGISTRY)/$(GPU_IMAGE)
	docker tag $(CPU_IMAGE) $(DOCKER_REGISTRY)/$(CPU_IMAGE)
	docker push $(DOCKER_REGISTRY)/$(GPU_IMAGE)
	docker push $(DOCKER_REGISTRY)/$(CPU_IMAGE)

status: ## Check service status
	@echo "📊 Service Status"
	@echo "================"
	@curl -s http://localhost:$(PORT)/ | jq '.' || echo "❌ Service not responding"
	@echo ""
	@docker ps --filter "name=whisper" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

benchmark: ## Run performance benchmark
	@echo "⚡ Running benchmark..."
	@for model in whisper-1-fast whisper-1 whisper-1-quality; do \
		echo "\nTesting $$model:"; \
		time curl -s -X POST "http://localhost:$(PORT)/v1/audio/transcriptions" \
			-H "Authorization: Bearer $(API_KEYS)" \
			-F "file=@examples/test.wav" \
			-F "model=$$model" \
			-o /dev/null; \
	done