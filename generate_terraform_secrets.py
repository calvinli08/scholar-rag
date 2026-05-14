import os
from textwrap import dedent

def env_to_secrets_tf():
    env_path = ".env.example"
    output_path = "infrastructure/secrets.tf"

    if not os.path.exists(env_path):
        print(f"Error: {env_path} not found.")
        return

    with open(env_path, "r") as f:
        lines = f.readlines()

    definitions = []

    for line in lines:
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue
        
        # Split at the first '=' to handle values that might contain '='
        if "=" not in line:
            continue
            
        key, raw_value = line.split("=", 1)
        key = key.strip()
        # Remove quotes if they exist in the .env file
        value = raw_value.strip().strip("'").strip('"')

        # Type Inference Logic
        if value.lower() in ["true", "false"]:
            tf_type = "bool"
            hcl_value = value.lower()
        elif value.isdigit():
            tf_type = "number"
            hcl_value = value
        else:
            tf_type = "string"
            hcl_value = f'"{value}"'

        # Generate the HCL Block
        block = (
            f'variable "{key}" {{\n'
            f'  type      = {tf_type}\n'
            f'  sensitive = true\n'
            f'  default   = {hcl_value}\n'
            f'}}\n'
        )
        definitions.append(block)

    with open(output_path, "w") as f:
        f.write("# Auto-generated from .env.example file\n\n")

        f.write(dedent("""
            resource "kubernetes_secret" "app_secrets" {
                metadata {
                   name      = "scholarrag-env"
                   namespace = kubernetes_namespace.scholarrag.metadata[0].name
                }
                type = "Opaque"
            }
        """))

        f.write("\n".join(definitions))
    
    print(f"Successfully created {output_path} with {len(definitions)} variable definitions.")

def write_tfvars_from_env():
    env_path=".env"
    output_path="infrastructure/secret.tfvars"

    if not os.path.exists(env_path):
        print(f"Error: {env_path} not found.")
        return

    with open(env_path, "r") as f:
        lines = f.readlines()

    assignments = []

    for line in lines:
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue
        
        if "=" not in line:
            continue
            
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip().strip("'").strip('"')

        # HCL Assignments in .tfvars handle types simply
        if value.lower() in ["true", "false"]:
            hcl_assignment = f'{key} = {value.lower()}'
        elif value.isdigit():
            hcl_assignment = f'{key} = {value}'
        else:
            # Escape backslashes or quotes if they exist in the value
            safe_value = value.replace('\\', '\\\\').replace('"', '\\"')
            hcl_assignment = f'{key} = "{safe_value}"'
            
        assignments.append(hcl_assignment)

    with open(output_path, "w") as f:
        f.write("# Environment Variable Assignments\n")
        f.write("# Auto-generated from .env file\n\n")
        f.write("# Use this with: terraform apply -var-file=\"secret.tfvars\"\n\n")
        f.write("\n".join(assignments))
    
    print(f"Successfully created {output_path} with {len(assignments)} variable assignments.")

if __name__ == "__main__":
    env_to_secrets_tf()
    write_tfvars_from_env()