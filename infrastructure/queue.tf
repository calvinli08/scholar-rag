resource "kubernetes_deployment" "queue" {
  metadata {
    name      = "scholarrag-queue"
    namespace = kubernetes_namespace.scholarrag.metadata[0].name
    labels = {
      app = "scholarrag-queue"
    }
  }

  spec {
    replicas = var.queue_replicas

    selector {
      match_labels = {
        app = "scholarrag-queue"
      }
    }

    template {
      metadata {
        labels = {
          app = "scholarrag-queue"
        }
      }

      spec {
        container {
          name  = "queue"
          image = var.queue_image

          env_from {
            secret_ref {
              name = kubernetes_secret.app_secrets.metadata[0].name
            }
          }

          resources {
            requests = {
              cpu    = "250m"
              memory = "512Mi"
            }
            limits = {
              cpu    = "500m"
              memory = "1Gi"
            }
          }
        }
      }
    }
  }
}

resource "kubernetes_horizontal_pod_autoscaler" "queue_hpa" {
  metadata {
    name      = "queue-hpa"
    namespace = kubernetes_namespace.scholarrag.metadata[0].name
  }

  spec {
    max_replicas = 2
    min_replicas = var.queue_replicas

    scale_target_ref {
      kind = "Deployment"
      name = kubernetes_deployment.queue.metadata[0].name
    }

    target_cpu_utilization_percentage = 80
  }
}