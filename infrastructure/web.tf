resource "kubernetes_deployment" "web" {
  metadata {
    name      = "scholarrag-web"
    namespace = kubernetes_namespace.scholarrag.metadata[0].name
    labels = {
      app = "scholarrag-web"
    }
  }

  spec {
    replicas = var.web_replicas

    selector {
      match_labels = {
        app = "scholarrag-web"
      }
    }

    template {
      metadata {
        labels = {
          app = "scholarrag-web"
        }
      }

      spec {
        container {
          name  = "web"
          image = var.web_image

          port {
            container_port = 8000
          }

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

resource "kubernetes_horizontal_pod_autoscaler" "web_hpa" {
  metadata {
    name      = "web-hpa"
    namespace = kubernetes_namespace.scholarrag.metadata[0].name
  }

  spec {
    max_replicas = 2
    min_replicas = var.web_replicas

    scale_target_ref {
      kind = "Deployment"
      name = kubernetes_deployment.web.metadata[0].name
    }

    target_cpu_utilization_percentage = 70
  }
}