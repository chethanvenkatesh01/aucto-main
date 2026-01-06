# --- Stage 1: Build the Cockpit ---
FROM node:18-alpine as build

WORKDIR /app

# Install Dependencies
COPY package*.json ./
# FIX: Use 'npm install' instead of 'npm ci' to auto-generate missing lockfile
RUN npm install

# Copy Source Code
COPY . .

# Compile for Production
RUN npm run build

# --- Stage 2: Serve via Nginx ---
FROM nginx:alpine

# Copy the build artifacts from Stage 1 to Nginx
COPY --from=build /app/dist /usr/share/nginx/html

# Default Nginx Config for React Router (SPA Fallback)
RUN echo 'server { \
    listen 80; \
    location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
