// Function to create and inject the button
function injectButton(tweet) {
    const actionBar = tweet.querySelector('[role="group"]');
    // Don't add if the button already exists or there's no action bar
    if (!actionBar || actionBar.querySelector('.twitter-analyze-btn')) return;

    const btn = document.createElement('button');
    btn.innerText = '🔍 Analyze';
    btn.className = 'twitter-analyze-btn';
    
    // Style to match your "Scientific Red" theme
    btn.style.cssText = `
        background: #FF4B4B; 
        color: white; 
        border: none; 
        border-radius: 999px;
        cursor: pointer; 
        margin-left: 12px; 
        padding: 4px 12px; 
        font-size: 13px; 
        font-weight: bold;
        transition: transform 0.1s ease;
    `;

    btn.onmouseover = () => btn.style.transform = "scale(1.05)";
    btn.onmouseout = () => btn.style.transform = "scale(1)";

    btn.onclick = (e) => {
        e.stopPropagation();
        e.preventDefault(); // Prevent Twitter from opening the tweet
        
        try {
            const textElement = tweet.querySelector('[data-testid="tweetText"]');
            const tweetText = textElement ? textElement.innerText : "No text found";
            
            const userElement = tweet.querySelector('[data-testid="User-Name"]');
            // Safely grab the @username
            const username = userElement ? userElement.innerText.split('\n')[1] || "Unknown" : "Unknown_User";
            
            // Send to your Streamlit App
            const url = `http://localhost:8501/?username=${encodeURIComponent(username)}&comment=${encodeURIComponent(tweetText)}`;
            window.open(url, '_blank'); 
        } catch (err) {
            console.error("Agent Scraper Error:", err);
        }
    };
    actionBar.appendChild(btn);
}

// Optimized Observer: Watches for new tweets added to the DOM
const observer = new MutationObserver((mutations) => {
    for (let mutation of mutations) {
        mutation.addedNodes.forEach(node => {
            if (node.nodeType === 1) { // Ensure it's an element
                // Check if the added node is a tweet or contains tweets
                const tweets = node.querySelectorAll?.('[data-testid="tweet"]');
                if (tweets) tweets.forEach(injectButton);
                if (node.getAttribute?.('data-testid') === 'tweet') injectButton(node);
            }
        });
    }
});

// Start watching the page
observer.observe(document.body, { childList: true, subtree: true });

// Initial run for tweets already on the page
document.querySelectorAll('[data-testid="tweet"]').forEach(injectButton);