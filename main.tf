terraform {
  required_providers {
    prefect = {
      source  = "PrefectHQ/prefect"
      version = ">= 0.1.0"
    }
  }
}

provider "prefect" {
  api_key   = var.prefect_api_key
}

resource "prefect_flow" "my_flow" {
  name       = "main"
  entrypoint = "pipeline.py:main"
}

resource "prefect_deployment" "my_deployment" {
  name         = "my-predictive-maintenance"
  flow_id      = prefect_flow.my_flow.id
  work_pool_id = "your-work-pool-id" # Replace with your actual work pool ID
}