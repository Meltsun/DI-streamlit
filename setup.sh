mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"18211199@bjtu.edu.cn\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml