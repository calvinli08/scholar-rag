variable "web_replicas" {
  description = "Initial number of replicas for the web deployment"
  type        = number
  default     = 1
}

variable "queue_replicas" {
  description = "Initial number of replicas for the queue deployment"
  type        = number
  default     = 1
}

variable "web_image" {
  description = "Docker image for the web component"
  type        = string
}

variable "queue_image" {
  description = "Docker image for the queue component"
  type        = string
}