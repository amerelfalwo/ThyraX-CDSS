#!/bin/bash
set -euo pipefail

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${CYAN}============================================${RESET}"
echo -e "${CYAN}  ThyraX CDSS - Azure VM Setup Script      ${RESET}"
echo -e "${CYAN}============================================${RESET}"

# Step 1: System Update & Upgrade
echo -e "\n${YELLOW}[1/5] Updating system packages...${RESET}"
sudo apt-get update -y
sudo apt-get upgrade -y

# Step 2: Install Docker prerequisites
echo -e "\n${YELLOW}[2/5] Installing Docker dependencies...${RESET}"
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Step 3: Set up Docker's official GPG key and apt repository
echo -e "\n${YELLOW}[3/5] Configuring official Docker apt repository...${RESET}"
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y

# Install Docker Engine, CLI, containerd, and Compose plugin
echo -e "\n${YELLOW}[4/5] Installing Docker Engine and Compose plugin...${RESET}"
sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# Enable and start Docker service
sudo systemctl enable docker
sudo systemctl start docker

# Step 4: Add current user to docker group (no sudo needed)
echo -e "\n${YELLOW}[5/5] Adding current user (${USER}) to the docker group...${RESET}"
sudo usermod -aG docker "$USER"

# Step 5: Create a .env template in the project directory
PROJECT_DIR="$(dirname "$(realpath "$0")")/.."
ENV_FILE="$PROJECT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo -e "\n${YELLOW}Creating .env template...${RESET}"
    cat > "$ENV_FILE" <<EOF
# Google AI Studio — Discrete keys per service
GOOGLE_API_KEY_LABS=your_google_ai_key_here
GOOGLE_API_KEY_VISION=your_google_ai_key_here
GOOGLE_API_KEY_AGENT=your_google_ai_key_here

# Database
DATABASE_URL=sqlite+aiosqlite:///./thyrax.db

# Internal Microservice Auth
INTERNAL_SERVICE_KEY=change_me_to_a_strong_random_secret

# ChromaDB
CHROMA_PATH=./data/vector_store
EOF
    echo -e "${GREEN}.env template created at: $ENV_FILE${RESET}"
else
    echo -e "${CYAN}.env file already exists. Skipping template creation.${RESET}"
fi

# Done — Print next steps
echo -e "\n${GREEN}============================================${RESET}"
echo -e "${GREEN}  Docker installed successfully!            ${RESET}"
echo -e "${GREEN}============================================${RESET}"

echo -e "\n${CYAN}NEXT STEPS:${RESET}"
echo -e "  ${YELLOW}1.${RESET} Log out and back in (or run ${CYAN}newgrp docker${RESET}) to activate Docker group."
echo -e "  ${YELLOW}2.${RESET} Edit your API keys: ${CYAN}nano $ENV_FILE${RESET}"
echo -e "  ${YELLOW}3.${RESET} Navigate to the project root: ${CYAN}cd $PROJECT_DIR${RESET}"
echo -e "  ${YELLOW}4.${RESET} Launch the full cluster: ${CYAN}docker compose up --build -d${RESET}"
echo -e "\n${CYAN}Service URLs (once running):${RESET}"
echo -e "  ${GREEN}FastAPI Backend  →${RESET} http://<your-vm-ip>:8000"
echo -e "  ${GREEN}Streamlit UI     →${RESET} http://<your-vm-ip>:8501"
echo -e "  ${GREEN}MLflow Tracking  →${RESET} http://<your-vm-ip>:5000"
echo -e "\n${RED}Azure Reminder:${RESET} Open ports 8000, 8501, and 5000 in your NSG Inbound Security Rules."
echo -e "\n${GREEN}Done!${RESET}"
