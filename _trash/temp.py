# Tắt tất cả các tunnel ngrok hiện tại
from pyngrok import ngrok

# Tắt tất cả tunnels
ngrok.kill()

# Hoặc tắt từng tunnel cụ thể
tunnels = ngrok.get_tunnels()
for tunnel in tunnels:
    ngrok.disconnect(tunnel.public_url)