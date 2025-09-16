import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import axios from "axios";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const HeroSection = () => {
  const [activeTab, setActiveTab] = useState("text");
  const navigate = useNavigate();
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const API_BASE = import.meta.env.VITE_API_BASE_URL as string;

  const renderInputField = () => {
    switch (activeTab) {
      case "text":
        return (
          <textarea 
            id="tl-input-text" 
            placeholder="Enter suspicious message, news article, or claim…" 
            className="w-full min-h-[120px] p-4 border border-input rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-ring bg-background" 
            value={text} 
            onChange={(e) => setText(e.target.value)} 
          />
        );
      case "url":
        return (
          <input 
            id="tl-input-url" 
            type="url" 
            placeholder="Paste URL of article, post, or content to verify…" 
            className="w-full p-4 border border-input rounded-lg focus:outline-none focus:ring-2 focus:ring-ring bg-background" 
            value={url} 
            onChange={(e) => setUrl(e.target.value)} 
          />
        );
      case "file":
        return (
          <div id="tl-input-file" className="w-full p-8 border-2 border-dashed border-input rounded-lg text-center">
            <p className="text-muted-foreground">Upload image or document for analysis</p>
            <p className="text-sm text-muted-foreground mt-2">Drag & drop or click to browse</p>
          </div>
        );
      default:
        return null;
    }
  };

  const handleAnalyze = async () => {
    console.log("handleAnalyze triggered", { activeTab, text, url });

    try {
      if (activeTab === "text" && !text.trim()) { 
        alert("Please enter some text."); 
        return; 
      }
      if (activeTab === "url" && !url.trim()) { 
        alert("Please enter a URL."); 
        return; 
      }
      if (activeTab === "file") { 
        alert("File upload will be added next."); 
        return; 
      }

      setLoading(true);

      // FIXED: Send JSON body matching your new backend VerifyRequest model
      const requestBody = {
        content_type: activeTab, // "text" or "url"
        content: activeTab === "text" ? text : url,
        language: "en"
      };

      const analyzePath = `${API_BASE}/api/v1/verify`;
      
      console.log("Calling POST:", analyzePath, requestBody);
      
      const res = await axios.post(analyzePath, requestBody, {
        headers: { "Content-Type": "application/json" }
      });

      const data = res.data;
      const confidence_percent = Math.round((data?.confidence_score || 0) * 100);
      const navState = { 
        ...data, 
        confidence_percent,
        content: activeTab === "text" ? text : url,  // ADD THIS LINE
        text: activeTab === "text" ? text : "",      // ADD THIS LINE
        url: activeTab === "url" ? url : ""          // ADD THIS LINE
      };
      
      setResult(navState);
      
      navigate(
        { pathname: "/results", search: `?analysis_id=${data.analysis_id || ""}` },
        { state: navState }
      );
      console.log("Verify response:", res.data);
    } catch (err: any) {
      console.error(err);
      alert(`Error: ${err?.response?.status || ""} ${err?.message || "Request failed"}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="relative py-20 lg:py-24 overflow-hidden">
      {/* Gradient Background */}
      <div 
        className="absolute inset-0 bg-gradient-to-br from-truthlens-primary to-truthlens-secondary"
        style={{
          background: "var(--truthlens-gradient)"
        }}
      />
      
      <div className="relative z-10 container mx-auto px-4 lg:px-6">
        <div className="max-w-4xl mx-auto text-center text-white">
          {/* Main Headlines */}
          <div className="mb-8">
            <h1 className="text-3xl md:text-5xl lg:text-6xl font-bold mb-4 leading-tight">
              See Reality as It Is
            </h1>
            <h2 className="text-2xl md:text-3xl lg:text-4xl font-semibold mb-6 text-orange-200">
              Empowering India with unbiased truth
            </h2>
            <p className="text-lg md:text-xl text-blue-100 mb-8">
              AI-powered fact-checking for every Indian citizen
            </p>
          </div>

          {/* Stats Row */}
          <div className="flex flex-col md:flex-row justify-center gap-8 md:gap-16 mb-12">
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold">10K+</div>
              <div className="text-sm md:text-base text-blue-200">Content Analyzed</div>
            </div>
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold">95%</div>
              <div className="text-sm md:text-base text-blue-200">Accuracy Rate</div>
            </div>
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold">24/7</div>
              <div className="text-sm md:text-base text-blue-200">AI Monitoring</div>
            </div>
          </div>

          {/* Input Card */}
          <div className="max-w-2xl mx-auto bg-card rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-semibold text-card-foreground mb-4">Start Your Fact-Check</h3>
            
            {/* Tabs */}
            <div className="flex bg-secondary rounded-lg p-1 mb-4">
              <button
                onClick={() => setActiveTab("text")}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                  activeTab === "text"
                    ? "bg-primary text-primary-foreground shadow-sm"
                    : "text-secondary-foreground hover:bg-secondary/80"
                }`}
                aria-selected={activeTab === "text"}
              >
                Text
              </button>
              <button
                onClick={() => setActiveTab("url")}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                  activeTab === "url"
                    ? "bg-primary text-primary-foreground shadow-sm"
                    : "text-secondary-foreground hover:bg-secondary/80"
                }`}
                aria-selected={activeTab === "url"}
              >
                URL
              </button>
              <button
                onClick={() => setActiveTab("file")}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                  activeTab === "file"
                    ? "bg-primary text-primary-foreground shadow-sm"
                    : "text-secondary-foreground hover:bg-secondary/80"
                }`}
                aria-selected={activeTab === "file"}
              >
                File
              </button>
            </div>

            {/* Input Field */}
            <div className="mb-6">
              {renderInputField()}
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <Button
                size="lg"
                className="bg-primary hover:bg-primary-hover text-primary-foreground"
                onClick={handleAnalyze}
                disabled={loading}
              >
                {loading ? "Analyzing..." : "Start Fact-Checking"}
              </Button>
              <Button variant="outline" size="lg" className="border-2 border-card-foreground/20 text-card-foreground hover:bg-card-foreground hover:text-card">
                Learn More
              </Button>
            </div>

            {/* Result Card */}
            {result && (
              <Card className="mt-6 text-left">
                <CardHeader>
                  <CardTitle className="flex items-center gap-3">
                    <span>Analysis Result</span>
                    <span
                      className={`px-3 py-1 rounded-full text-sm font-semibold ${
                        result.verdict === "true"
                          ? "bg-green-500/10 text-green-600"
                          : result.verdict === "false"
                          ? "bg-red-500/10 text-red-600"
                          : "bg-amber-500/10 text-amber-600"
                      }`}
                    >
                      {(result.verdict || "inconclusive").toUpperCase()}
                    </span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="text-sm text-muted-foreground">
                    Confidence: <span className="font-semibold text-foreground">{result.confidence_percent}%</span>
                  </div>
                  {result.summary && (
                    <p className="text-foreground">{result.summary}</p>
                  )}
                  {result.analysis_id && (
                    <div className="text-xs text-muted-foreground">
                      ID: {result.analysis_id}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
