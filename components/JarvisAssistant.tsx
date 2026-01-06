import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, Send, X, Bot, Loader2, Sparkles } from 'lucide-react';
import { sendMessageToJarvis, ChatMessage } from '../services/geminiService';

interface JarvisProps {
  contextData: any;
}

const JarvisAssistant: React.FC<JarvisProps> = ({ contextData }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'model', text: 'Hello, I am Jarvis. I can explain the rationale behind any ADOS decision or anomaly. How can I assist?' }
  ]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg: ChatMessage = { role: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    const responseText = await sendMessageToJarvis(input, messages, contextData);
    
    setMessages(prev => [...prev, { role: 'model', text: responseText }]);
    setLoading(false);
  };

  return (
    <>
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 w-14 h-14 bg-gradient-to-br from-indigo-600 to-purple-700 text-white rounded-full shadow-[0_8px_30px_rgb(79,70,229,0.4)] flex items-center justify-center transition-all duration-300 z-50 hover:scale-110 group"
        >
          <Sparkles size={24} className="group-hover:rotate-12 transition-transform" />
          <span className="absolute inset-0 rounded-full animate-ping bg-indigo-500 opacity-20"></span>
        </button>
      )}

      <div 
        className={`fixed bottom-6 right-6 w-[400px] bg-white/90 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/40 flex flex-col transition-all duration-500 z-50 overflow-hidden ring-1 ring-slate-900/5 ${
          isOpen ? 'h-[650px] opacity-100 translate-y-0 scale-100' : 'h-0 opacity-0 translate-y-20 scale-95 pointer-events-none'
        }`}
      >
        <div className="bg-gradient-to-r from-indigo-600 to-purple-700 p-5 flex items-center justify-between text-white shrink-0 shadow-lg">
          <div className="flex items-center space-x-3">
            <div className="bg-white/20 p-2 rounded-xl backdrop-blur-md shadow-inner border border-white/10">
              <Bot size={22} />
            </div>
            <div>
              <h3 className="font-bold text-base tracking-tight">Jarvis AI</h3>
              <p className="text-[10px] text-indigo-100 font-medium uppercase tracking-widest opacity-80">Online • v2.4</p>
            </div>
          </div>
          <button onClick={() => setIsOpen(false)} className="hover:bg-white/20 p-2 rounded-full transition-colors">
            <X size={20} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-6 bg-slate-50/50">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in slide-in-from-bottom-2 duration-300`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-5 py-3.5 text-sm shadow-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-br from-indigo-600 to-indigo-700 text-white rounded-br-sm'
                    : 'bg-white text-slate-800 border border-slate-100 rounded-bl-sm shadow-md'
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start animate-pulse">
              <div className="bg-white rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm border border-slate-100 flex items-center space-x-2">
                <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
                <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
                <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 bg-white/80 border-t border-slate-100 shrink-0 backdrop-blur-sm">
          <div className="flex items-center space-x-2 bg-slate-100 rounded-full px-2 py-2 border border-slate-200 focus-within:border-indigo-400 focus-within:ring-4 focus-within:ring-indigo-100 transition-all shadow-inner">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask about data trends..."
              className="flex-1 bg-transparent border-none outline-none text-sm text-slate-700 placeholder:text-slate-400 px-3"
            />
            <button 
              onClick={handleSend}
              disabled={loading || !input.trim()}
              className="w-9 h-9 flex items-center justify-center rounded-full bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50 transition-all shadow-md hover:scale-105"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>
    </>
  );
};

export default JarvisAssistant;