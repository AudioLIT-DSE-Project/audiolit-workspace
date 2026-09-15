# LIT-203 — production frontend image.
# Build context is the repo root (shares .dockerignore with the backend).
# VITE_API_BASE_URL is baked in at build time (see Frontend/src/lib/api.ts);
# compose overrides it with http://127.0.0.1:8000 so the session cookie stays
# same-site (SameSite=Lax) across api:8000 <-> web:8080.
FROM node:20 AS build
WORKDIR /build
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
COPY Frontend/package.json Frontend/package-lock.json /build/
RUN npm ci
COPY Frontend/ /build/
RUN npm run build

FROM nginx:alpine
# apk upgrade hardens the base tag against freshly-published CRITICALs
# (Trivy gate, LIT-203); same release, no pin drift.
RUN apk upgrade --no-cache
COPY Frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /build/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8080/ > /dev/null || exit 1