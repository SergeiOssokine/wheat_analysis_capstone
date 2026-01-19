
# Self-documenting Makefile, taken from
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

IMG_NAME =  grain_prediction_service

get_dataset: ## Download the seed dataset from UCI
	cd data && bash get_data.sh
	
setup_env: ## Set up the local environment by installing the required Python packages
	bash setup_env.sh

train: ## Train the best model (LinearDiscriminantAnalysis)
	uv run python train.py

build_prediction_service:  ## Build Docker image with the model running as a FastAPI webservice
	docker build -t ${IMG_NAME}:v0.1 .

serve_predictions: ## Launch the prediction service container
	docker run -p 9696:9696 -d --name ${IMG_NAME} ${IMG_NAME}:v0.1

test_prediction_service: ## Run an example inference
	uv run test_prediction_service.py
	
shutdown_prediction_service: ## Stop the prediction service docker container
	docker kill ${IMG_NAME}

remove_prediction_container: ## Remove the prediction service container
	docker container rm ${IMG_NAME}

cleanup_venv: ## Remove the Python virtual environment created with setup_env
	rm -rf ./.venv

help:
	@echo "Possible actions:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'