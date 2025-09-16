import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import TruthLensHeader from "@/components/TruthLensHeader";
import TruthLensFooter from "@/components/TruthLensFooter";

const Learn = () => {
  const [activeTab, setActiveTab] = useState("overview");

  const tabs = [
    { id: "overview", icon: "📚", label: "Overview" },
    { id: "source", icon: "🔍", label: "Source Verification" },
    { id: "image", icon: "🖼️", label: "Image Analysis" },
    { id: "data", icon: "📊", label: "Data Verification" },
    { id: "emotional", icon: "💭", label: "Emotional Manipulation" },
    { id: "test", icon: "🎯", label: "Test Your Skills" }
  ];

  const stats = [
    { icon: "📈", number: "89%", label: "Success Rate After Training" },
    { icon: "👥", number: "25K+", label: "Users Trained" },
    { icon: "🎓", number: "12", label: "Learning Modules" },
    { icon: "⏱️", number: "30min", label: "Average Completion Time" }
  ];

  const misinformationTypes = [
    {
      icon: "🔄",
      title: "Recycled Content",
      description: "Old content presented as new or current events",
      examples: "Historical photos used for recent events"
    },
    {
      icon: "🎭",
      title: "Manipulated Media",
      description: "Edited images, videos, or audio content",
      examples: "Deepfakes, photo editing, voice synthesis"
    },
    {
      icon: "📰",
      title: "False Context",
      description: "Real content shared with misleading information",
      examples: "Correct image with wrong location or date"
    },
    {
      icon: "🔮",
      title: "Fabricated Content",
      description: "Completely fake information created from scratch",
      examples: "Made-up statistics, fictional quotes"
    }
  ];

  const skillLevels = [
    { level: "Beginner", progress: 100, description: "Basic misinformation recognition" },
    { level: "Intermediate", progress: 75, description: "Advanced verification techniques" },
    { level: "Expert", progress: 45, description: "Complex analysis and fact-checking" }
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case "overview":
        return (
          <div className="space-y-8">
            <div>
              <h3 className="text-2xl font-bold text-foreground mb-4">Understanding Misinformation</h3>
              <p className="text-muted-foreground mb-6 leading-relaxed">
                Misinformation is false or inaccurate information that is spread regardless of intent. 
                In the digital age, it spreads faster than ever before, making fact-checking skills essential 
                for every internet user.
              </p>
              
              <div className="grid md:grid-cols-4 gap-6 mb-8">
                {stats.map((stat, index) => (
                  <div key={index} className="text-center">
                    <div className="text-3xl mb-2">{stat.icon}</div>
                    <div className="text-2xl font-bold text-truthlens-primary mb-1">
                      {stat.number}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {stat.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-xl font-semibold text-foreground mb-6">Types of Misinformation</h3>
              <div className="grid md:grid-cols-2 gap-6">
                {misinformationTypes.map((type, index) => (
                  <Card key={index} className="hover:shadow-md transition-shadow">
                    <CardContent className="p-6">
                      <div className="flex items-start gap-3 mb-3">
                        <div className="text-2xl">{type.icon}</div>
                        <div>
                          <h4 className="font-semibold text-card-foreground">{type.title}</h4>
                        </div>
                      </div>
                      <p className="text-muted-foreground text-sm mb-3">
                        {type.description}
                      </p>
                      <p className="text-xs text-muted-foreground italic">
                        <strong>Examples:</strong> {type.examples}
                      </p>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </div>
        );
      
      case "test":
        return (
          <div className="space-y-8">
            <div>
              <h3 className="text-2xl font-bold text-foreground mb-4">Test Your Skills</h3>
              <p className="text-muted-foreground mb-6">
                Practice identifying misinformation with our interactive challenges.
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-6 mb-8">
              {skillLevels.map((skill, index) => (
                <Card key={index}>
                  <CardContent className="p-6">
                    <div className="flex justify-between items-center mb-3">
                      <h4 className="font-semibold text-card-foreground">{skill.level}</h4>
                      <span className="text-sm text-muted-foreground">{skill.progress}%</span>
                    </div>
                    <div className="w-full bg-secondary/20 rounded-full h-2 mb-3">
                      <div 
                        className="h-2 rounded-full bg-truthlens-primary"
                        style={{ width: `${skill.progress}%` }}
                      />
                    </div>
                    <p className="text-sm text-muted-foreground mb-4">
                      {skill.description}
                    </p>
                    <Button variant="outline" size="sm" className="w-full">
                      {skill.progress === 100 ? "Review" : "Continue"}
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>

            <Card className="bg-secondary/10">
              <CardContent className="p-6 text-center">
                <h4 className="text-xl font-semibold text-foreground mb-3">
                  Ready for the Challenge?
                </h4>
                <p className="text-muted-foreground mb-4">
                  Take our comprehensive fact-checking assessment and earn your certification.
                </p>
                <Button size="lg">
                  Start Assessment
                </Button>
              </CardContent>
            </Card>
          </div>
        );
      
      default:
        return (
          <div className="text-center py-12">
            <div className="text-4xl mb-4">🚧</div>
            <h3 className="text-xl font-semibold text-foreground mb-2">Coming Soon</h3>
            <p className="text-muted-foreground">
              This learning module is currently being developed.
            </p>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <TruthLensHeader />
      <main className="flex-1">
        {/* Hero Section */}
        <section className="relative py-16 lg:py-20 overflow-hidden">
          <div 
            className="absolute inset-0 opacity-10"
            style={{ background: "var(--truthlens-gradient)" }}
          />
          <div className="container relative mx-auto px-4 lg:px-6">
            <div className="text-center mb-12">
              <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-6">
                Learn to Identify Misinformation
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
                Educational resources to become a digital fact-checker
              </p>
            </div>
          </div>
        </section>

        {/* Navigation Tabs */}
        <section className="py-8 bg-card border-b border-border">
          <div className="container mx-auto px-4 lg:px-6">
            <div className="flex flex-wrap gap-2 justify-center">
              {tabs.map((tab) => (
                <Button
                  key={tab.id}
                  variant={activeTab === tab.id ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setActiveTab(tab.id)}
                  className="flex items-center gap-2"
                >
                  <span>{tab.icon}</span>
                  <span className="hidden sm:inline">{tab.label}</span>
                </Button>
              ))}
            </div>
          </div>
        </section>

        {/* Content Sections */}
        <section className="py-12">
          <div className="container mx-auto px-4 lg:px-6">
            <div className="max-w-4xl mx-auto">
              {renderTabContent()}
            </div>
          </div>
        </section>
      </main>
      <TruthLensFooter />
    </div>
  );
};

export default Learn;