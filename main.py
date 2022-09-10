from pyngrok import ngrok

# Open a HTTP tunnel on the default port 80
# <NgrokTunnel: "http://<public_sub>.ngrok.io" -> "http://localhost:80">
http_tunnel = ngrok.connect()


