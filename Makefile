# Voice Agent Docker Management
.PHONY: help build build-dev up up-dev down down-dev logs clean restart health

# Default target
help:
	@echo "Voice Agent Docker Commands:"
	@echo ""
	@echo "Production:"
	@echo "  make build     - Build all production containers"
	@echo "  make up        - Start production environment"
	@echo "  make down      - Stop production environment"
	@echo ""
	@echo "Development:"
	@echo "  make build-dev - Build development containers"
	@echo "  make up-dev    - Start development environment"
	@echo "  make down-dev  - Stop development environment"
	@echo ""
	@echo "Management:"
	@echo "  make logs      - Show logs for all services"
	@echo "  make health    - Check health of all services"
	@echo "  make restart   - Restart all services"
	@echo "  make clean     - Remove all containers and volumes"
	@echo ""
	@echo "Setup:"
	@echo "  make setup     - Initial setup with example env file"
	@echo "  make install   - Install dependencies locally"
	@echo ""
	@echo "Testing:"
	@echo "  make test      - Run all tests"
	@echo "  make test-personal-facts - Test personal fact storage pipeline"

# Production commands
build:
	@echo "🏗️  Building production containers..."
	docker-compose build --parallel

up:
	@echo "🚀 Starting production environment..."
	docker-compose up -d
	@echo "✅ Production environment started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend API: http://localhost:8000"
	@echo "Nginx Proxy: http://localhost:80"

down:
	@echo "⏹️  Stopping production environment..."
	docker-compose down

# Development commands
build-dev:
	@echo "🏗️  Building development containers..."
	docker-compose -f docker-compose.dev.yml build --parallel

up-dev:
	@echo "🚀 Starting development environment..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "✅ Development environment started!"
	@echo "Backend API: http://localhost:8000"
	@echo "Database: localhost:5432"
	@echo "Redis: localhost:6379"

down-dev:
	@echo "⏹️  Stopping development environment..."
	docker-compose -f docker-compose.dev.yml down

# Management commands
logs:
	@echo "📋 Showing logs for all services..."
	docker-compose logs -f

logs-dev:
	@echo "📋 Showing development logs..."
	docker-compose -f docker-compose.dev.yml logs -f

health:
	@echo "🏥 Checking service health..."
	@echo "Frontend Health:"
	@curl -s http://localhost:3000/api/health | jq . || echo "Frontend not responding"
	@echo ""
	@echo "Backend Health:"
	@curl -s http://localhost:8000/health | jq . || echo "Backend not responding"
	@echo ""
	@echo "Database Health:"
	@docker-compose exec postgres pg_isready -U voice_agent || echo "Database not responding"

restart:
	@echo "🔄 Restarting all services..."
	docker-compose restart

restart-dev:
	@echo "🔄 Restarting development services..."
	docker-compose -f docker-compose.dev.yml restart

clean:
	@echo "🧹 Cleaning up containers and volumes..."
	docker-compose down -v --remove-orphans
	docker-compose -f docker-compose.dev.yml down -v --remove-orphans
	docker system prune -f
	@echo "✅ Cleanup complete!"

# Setup commands
setup:
	@echo "⚙️  Setting up Voice Agent..."
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env file from example..."; \
		cp env.example .env; \
		echo "Please edit .env file with your configuration"; \
	else \
		echo ".env file already exists"; \
	fi
	@mkdir -p logs uploads ssl
	@echo "✅ Setup complete! Please edit .env file before running make up"

install:
	@echo "📦 Installing dependencies..."
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Installing Node.js dependencies..."
	cd react-frontend && npm install --legacy-peer-deps
	@echo "✅ Dependencies installed!"

# Database commands
db-shell:
	@echo "🐘 Connecting to database..."
	docker-compose exec postgres psql -U voice_agent -d voice_agent

db-backup:
	@echo "💾 Creating database backup..."
	docker-compose exec postgres pg_dump -U voice_agent voice_agent > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Backup created!"

db-restore:
	@echo "📥 Restoring database..."
	@read -p "Enter backup file path: " backup_file; \
	docker-compose exec -T postgres psql -U voice_agent -d voice_agent < $$backup_file

# Development helpers
frontend-shell:
	@echo "🖥️  Connecting to frontend container..."
	docker-compose exec frontend sh

backend-shell:
	@echo "🔧 Connecting to backend container..."
	docker-compose exec backend bash

# Testing
test:
	@echo "🧪 Running tests..."
	docker-compose exec backend python -m pytest
	docker-compose exec frontend npm test

test-personal-facts:
	@echo "🧪 Testing Personal Facts Pipeline..."
	python test_personal_facts.py

# Security scan
security-scan:
	@echo "🔒 Running security scan..."
	docker run --rm -v $(PWD):/app clair-scanner:latest /app 