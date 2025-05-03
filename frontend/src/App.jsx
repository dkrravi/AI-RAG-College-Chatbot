import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [inputText, setInputText] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  const handleSendMessage = async () => {
    if (inputText.trim() === '') return;

    const userMessage = { text: inputText, sender: 'user' };
    setMessages([...messages, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/api/chat', {
        query: inputText,
      });

      const botMessage = { text: response.data.answer, sender: 'bot' };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const botMessage = { text: "I'm sorry, I couldn't get an answer.", sender: 'bot' };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div 
  className="container-fluid d-flex justify-content-center align-items-center" 
  style={{
    backgroundImage: "url('/BG1.png')",
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    backgroundRepeat: 'no-repeat',
    width: '100vw',  
    height: '100vh', 
    position: 'fixed', 
    top: 0,
    left: 0
  }}
>
<div 
  className="card shadow-lg" 
  style={{ 
    width: '580px',
    height: '600px', 
    background: 'rgba(255, 255, 255, 0.85)', 
    borderRadius: '10px',
    marginLeft: '-200px'
  }}
>
        <div className="card-header text-center bg-primary text-white">
          <h3>NGPCAS Assist </h3>
        </div>
        <div className="card-body overflow-auto" style={{ height: 'calc(100% - 120px)' }}>
          <div className="d-flex flex-column">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`p-2 my-2 rounded ${
                  message.sender === 'user'
                    ? 'bg-primary text-white align-self-end text-end'
                    : 'bg-light text-dark align-self-start text-justify'
                }`}
                style={{
                  maxWidth: '75%',
                  textAlign: message.sender === 'user' ? 'right' : 'justify',
                }}
              >
                {message.text}
              </div>
            ))}
            {isLoading && (
              <div
                className="p-2 my-2 rounded bg-light text-muted align-self-start text-start"
                style={{ maxWidth: '85%', textAlign: 'left' }}
              >
                Typing...
              </div>
            )}
          </div>
        </div>
        <div className="card-footer d-flex">
          <input
            type="text"
            className="form-control me-2"
            value={inputText}
            onChange={handleInputChange}
            onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Type your message..."
          />
          <button onClick={handleSendMessage} className="btn btn-primary">
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
