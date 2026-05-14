terraform {
    required_version = ">= 1.15.3"
    required_providers {
      kubernetes = {
        source = "hashicorp/kubernetes"
        version = ">= 3.1.0"
      }
    }
}

provider "kubernetes" {
    config_path = "~/.kube/config"
    config_context = "docker-desktop"
}

resource "kubernetes_namespace" "scholarrag" {
  metadata {
    name = "scholarrag"
  }
}