# Build and Push DGE Docker Image
param(
    [string]$ImageName = "darealsven/dge-env",
    [string]$Version = "v0.1.0"
)

Write-Host "🐳 Building Docker Image: $ImageName:$Version" -ForegroundColor Cyan

# Build
docker build -t "$ImageName`:$Version" -t "$ImageName`:latest" .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build Successful!" -ForegroundColor Green
    
    Write-Host "🚀 Pushing to Docker Hub..." -ForegroundColor Cyan
    docker push "$ImageName`:$Version"
    docker push "$ImageName`:latest"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Push Successful!" -ForegroundColor Green
    } else {
        Write-Host "❌ Push Failed. Are you logged in? (docker login)" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Build Failed." -ForegroundColor Red
}
