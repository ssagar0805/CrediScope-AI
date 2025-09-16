import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { useNavigate, useLocation, Navigate } from "react-router-dom";
import {
  ArrowLeft,
  Eye,
  FileText,
  Brain,
  GraduationCap,
  Search,
  Star,
  CheckCircle,
  Clock,
  Loader2,
  Download,
  Flag,
  Share2,
} from "lucide-react";
import TruthLensHeader from "@/components/TruthLensHeader";
import TruthLensFooter from "@/components/TruthLensFooter";
import { streamAnalysis } from "@/api/client";

// 1. Update interface to fetch the educational block from backend
interface LocationState {
  content?: string;
  text?: string;
  url?: string;
  verdict?: string;
  confidence_score?: number;
  confidence_percent?: number;
  summary?: string;
  analysis_id?: string;
  detailed_analysis?: any;
  educational?: any;
}

const Results = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const state = (location.state as LocationState) || {};

  // Redirect if no state
  if (!state || Object.keys(state).length === 0) {
    return <Navigate to="/" replace />;
  }

  // 🚀 NEW: Streaming state
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState<string>("");
  const [streamingData, setStreamingData] = useState<any>({});
  const [streamingError, setStreamingError] = useState<string>("");
  
  // Original state
  const [viewMode, setViewMode] = useState<"summary" | "full">("summary");
  const [expandedCards, setExpandedCards] = useState<string[]>([]);

  // Get content to analyze
  const contentToAnalyze = state.content ?? state.text ?? state.url ?? "";

  // 🚀 NEW: Start streaming when viewMode switches to "full"
  useEffect(() => {
    if (viewMode === "full" && !isStreaming && contentToAnalyze && !streamingData.misinformation_analysis) {
      startLiveAnalysis();
    }
  }, [viewMode, contentToAnalyze]);

  const startLiveAnalysis = async () => {
    if (!contentToAnalyze.trim()) return;
    
    setIsStreaming(true);
    setStreamingError("");
    setStreamingData({});

    try {
      await streamAnalysis(
        "/api/v1/verify-stream",
        {
          content_type: "text",
          content: contentToAnalyze,
          language: "en"
        },
        // onMessage callback
        (data: any) => {
          console.log("🔴 Streaming data:", data);
          
          if (data.type === "message") {
            setStreamingMessage(data.content);
          }
          
          if (data.type === "section") {
            setStreamingData((prev: any) => ({
              ...prev,
              [data.section]: data.data
            }));
          }
          
          if (data.type === "complete") {
            setIsStreaming(false);
            setStreamingMessage("✅ Analysis complete!");
          }
        },
        // onError callback
        (error: Error) => {
          console.error("🚨 Streaming error:", error);
          setStreamingError(`Connection failed: ${error.message}`);
          setIsStreaming(false);
        },
        // onComplete callback
        () => {
          setIsStreaming(false);
        }
      );
    } catch (error: any) {
      console.error("🚨 Stream start error:", error);
      setStreamingError(`Failed to start analysis: ${error.message}`);
      setIsStreaming(false);
    }
  };

  // 3. Map backend data to frontend variables (fallback to static data)
  const educationalData = streamingData.misinformation_analysis 
    ? streamingData 
    : (state.educational || 
       (state.detailed_analysis && state.detailed_analysis.educational) || 
       {});

  // 🔧 FIXED: Content extraction functions for nested backend structure
  const extractEducationalContent = (data: any, section: string): React.ReactNode => {
    if (!data || typeof data !== 'object') {
      return isStreaming ? (
        <div className="flex items-center gap-2 text-muted-foreground py-4">
          <LoadingAnimation />
          <span className="ml-2">Waiting for {section.replace('_', ' ')}...</span>
        </div>
      ) : "Waiting for analysis...";
    }

    const sectionData = data[section];
    if (!sectionData || typeof sectionData !== 'object') {
      return isStreaming ? (
        <div className="flex items-center gap-2 text-muted-foreground py-4">
          <LoadingAnimation />
          <span className="ml-2">Generating content...</span>
        </div>
      ) : "Waiting for analysis...";
    }

    // Convert object to readable text with better formatting
    const content = Object.entries(sectionData)
      .filter(([key, value]) => key !== 'raw_response' && typeof value === 'string')
      .map(([key, value]) => (
        <div key={key} className="mb-4">
          <h5 className="font-semibold text-primary mb-2 capitalize">
            {key.replace(/_/g, ' ')}:
          </h5>
          <div className="pl-4 border-l-2 border-primary/20">
            <p className="whitespace-pre-line text-sm leading-relaxed">{value}</p>
          </div>
        </div>
      ));

    return content.length > 0 ? <div>{content}</div> : "Waiting for analysis...";
  };

  const analysisData = {
    content: state.content ?? state.text ?? state.url,
    risk: state.verdict
      ? {
          level:
            state.verdict === "false"
              ? "High"
              : state.verdict === "true"
              ? "Low"
              : "Medium",
          score:
            state.confidence_percent ??
            Math.round((state.confidence_score ?? 0) * 100),
          confidence:
            state.confidence_percent ??
            Math.round((state.confidence_score ?? 0) * 100),
          verdict: state.verdict,
        }
      : undefined,
    recommendations: state.summary ? [state.summary] : [],
  };

  // 4. Educational accordion configs
  const educationalConfigs = [
    {
      id: "misinformation",
      title: "🧠 Why This is Misinformation",
      icon: Brain,
      content_path: "misinformation_analysis",
    },
    {
      id: "indians-know",
      title: "🎓 What Indians Should Know",
      icon: GraduationCap,
      content_path: "indian_context",
    },
    {
      id: "pattern-recognition",
      title: "🔍 How to Spot Similar Claims",
      icon: Search,
      content_path: "pattern_recognition",
    },
    {
      id: "real-story",
      title: "🌟 The Real Story",
      icon: Star,
      content_path: "real_story",
    },
  ];

  // 5. Loading animation
  const LoadingAnimation = () => (
    <div className="flex items-center gap-2 text-muted-foreground">
      <Loader2 className="w-4 h-4 animate-spin" />
      <span className="text-sm">Processing analysis...</span>
      <div className="flex gap-1">
        {[0, 150, 300].map((d) => (
          <div
            key={d}
            className="w-1 h-1 bg-primary rounded-full animate-bounce"
            style={{ animationDelay: `${d}ms` }}
          />
        ))}
      </div>
    </div>
  );

  // 6. Risk badge
  const getRiskStyling = (level?: string) => {
    switch (level) {
      case "High":
        return { bg: "bg-red-50", text: "text-red-700" };
      case "Medium":
        return { bg: "bg-amber-50", text: "text-amber-700" };
      case "Low":
        return { bg: "bg-green-50", text: "text-green-700" };
      default:
        return { bg: "bg-muted/20", text: "text-muted-foreground" };
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <TruthLensHeader />
      <main className="flex-1">
        <section className="bg-gradient-to-br from-primary to-secondary border-b border-border">
          <div className="container mx-auto px-4 lg:px-6">
            <div className="flex items-center justify-between py-6">
              <div className="flex items-center gap-4">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => navigate("/")}
                  className="text-white/90 hover:text-white hover:bg-white/10"
                >
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back
                </Button>
                <div>
                  <h1 className="text-2xl font-bold text-white">
                    Analysis Results
                  </h1>
                  <div className="flex items-center gap-2 mt-1">
                    {isStreaming ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin text-yellow-300" />
                        <span className="text-white/80 text-sm">
                          {streamingMessage || "Live Analysis..."}
                        </span>
                      </>
                    ) : analysisData.risk ? (
                      <>
                        <CheckCircle className="w-4 h-4 text-green-300" />
                        <span className="text-white/80 text-sm">
                          Analysis Complete
                        </span>
                      </>
                    ) : (
                      <>
                        <Clock className="w-4 h-4 text-white/60" />
                        <span className="text-white/80 text-sm">
                          Awaiting Data...
                        </span>
                      </>
                    )}
                  </div>
                </div>
              </div>
              <div className="flex gap-1 bg-white/10 rounded-lg p-1">
                <Button
                  size="sm"
                  variant={viewMode === "summary" ? "default" : "ghost"}
                  onClick={() => setViewMode("summary")}
                  className={
                    viewMode === "summary"
                      ? "bg-white text-primary"
                      : "text-white/90"
                  }
                >
                  <Eye className="w-4 h-4 mr-1" />
                  Quick Summary
                </Button>
                <Button
                  size="sm"
                  variant={viewMode === "full" ? "default" : "ghost"}
                  onClick={() => setViewMode("full")}
                  className={
                    viewMode === "full"
                      ? "bg-white text-primary"
                      : "text-white/90"
                  }
                >
                  <FileText className="w-4 h-4 mr-1" />
                  Live Analysis
                </Button>
              </div>
            </div>
          </div>
        </section>

        <div className="container mx-auto px-4 lg:px-6 py-8">
          <div className="max-w-4xl mx-auto space-y-6">
            {/* 🚀 NEW: Show streaming error if any */}
            {streamingError && (
              <Card className="border-red-200 bg-red-50">
                <CardContent className="pt-6">
                  <div className="text-red-700 text-sm">
                    <strong>Connection Error:</strong> {streamingError}
                    <Button
                      onClick={startLiveAnalysis}
                      className="ml-4"
                      size="sm"
                      variant="outline"
                    >
                      Retry Live Analysis
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {viewMode === "summary" ? (
              analysisData.risk ? (
                <Card className="border-l-4 border-primary">
                  <CardContent className="pt-6">
                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <h3 className="font-semibold mb-4">Quick Summary</h3>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <span className="text-sm">Risk Level:</span>
                            <Badge
                              className={`${getRiskStyling(
                                analysisData.risk.level
                              ).bg} ${
                                getRiskStyling(analysisData.risk.level).text
                              }`}
                            >
                              {analysisData.risk.level}
                            </Badge>
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-xs">Credibility</span>
                              <span className="text-xs font-bold">
                                {analysisData.risk.score}%
                              </span>
                            </div>
                            <Progress
                              value={analysisData.risk.score}
                              className="h-2"
                            />
                          </div>
                        </div>
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2">
                          Key Recommendation
                        </h4>
                        <p className="text-sm text-muted-foreground p-3 bg-muted/30 rounded-lg">
                          {analysisData.recommendations?.[0] ||
                            "Complete analysis to see recommendations"}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardContent className="pt-6 text-center">
                    <LoadingAnimation />
                    <p className="text-sm text-muted-foreground mt-4">
                      Switch to "Live Analysis" to see real-time results
                    </p>
                  </CardContent>
                </Card>
              )
            ) : (
              <>
                {/* 🚀 NEW: Live streaming status */}
                {isStreaming && streamingMessage && (
                  <Card className="border-blue-200 bg-blue-50">
                    <CardContent className="pt-6">
                      <div className="flex items-center gap-3 text-blue-700">
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span className="font-medium">{streamingMessage}</span>
                      </div>
                    </CardContent>
                  </Card>
                )}

                <Accordion
                  type="multiple"
                  value={expandedCards}
                  onValueChange={setExpandedCards}
                >
                  {educationalConfigs.map((cfg) => {
                    const IconComponent = cfg.icon;
                    const contentVal = extractEducationalContent(educationalData, cfg.content_path);
                    const hasContent = streamingData[cfg.content_path] && Object.keys(streamingData[cfg.content_path]).length > 0;

                    return (
                      <AccordionItem key={cfg.id} value={cfg.id}>
                        <Card>
                          <AccordionTrigger className="px-6 py-4">
                            <div className="flex items-center gap-3">
                              <IconComponent className="w-5 h-5 text-primary" />
                              <div className="font-semibold">{cfg.title}</div>
                              {!hasContent && isStreaming && (
                                <div className="ml-auto">
                                  <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                                </div>
                              )}
                            </div>
                          </AccordionTrigger>
                          <AccordionContent>
                            <div className="px-6 pb-4">
                              {contentVal}
                            </div>
                          </AccordionContent>
                        </Card>
                      </AccordionItem>
                    );
                  })}
                </Accordion>
              </>
            )}

            {/* Actions */}
            <Card className="mt-8">
              <CardHeader>
                <CardTitle className="text-center">Actions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <Button
                    onClick={() => navigate("/")}
                    variant="default"
                    className="w-full"
                  >
                    <Search className="w-4 h-4 mr-2" />
                    Analyze Another
                  </Button>
                  <Button variant="outline" className="w-full" disabled>
                    <Flag className="w-4 h-4 mr-2" />
                    Flag for Authority
                  </Button>
                  <Button variant="outline" className="w-full" disabled>
                    <Download className="w-4 h-4 mr-2" />
                    Download Report
                  </Button>
                  <Button variant="outline" className="w-full" disabled>
                    <Share2 className="w-4 h-4 mr-2" />
                    Share Results
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground text-center mt-4">
                  More actions available after integration
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
      <TruthLensFooter />
    </div>
  );
};

export default Results;
