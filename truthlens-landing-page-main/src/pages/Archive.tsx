import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import TruthLensHeader from "@/components/TruthLensHeader";
import TruthLensFooter from "@/components/TruthLensFooter";

const Archive = () => {
  const [selectedFilter, setSelectedFilter] = useState("All Cases");
  const [selectedCategory, setSelectedCategory] = useState("All");

  const stats = [
    { number: "3,247", label: "TOTAL CASES" },
    { number: "2,891", label: "✅ VERIFIED" },
    { number: "356", label: "❌ FALSE" }
  ];

  const filters = ["All Cases", "✅ Verified", "⚠️ Caution", "❌ False"];
  const categories = [
    { name: "All", icon: "📊" },
    { name: "Health", icon: "🏥" },
    { name: "Politics", icon: "🗳️" },
    { name: "Finance", icon: "💰" },
    { name: "Social", icon: "📱" }
  ];

  const cases = [
    {
      id: 1,
      category: "Health",
      icon: "🏥",
      date: "2024-01-15",
      title: "False claim about vaccine side effects spreading on WhatsApp",
      status: "❌ False",
      statusColor: "bg-red-500/10 text-red-600"
    },
    {
      id: 2,
      category: "Politics",
      icon: "🗳️", 
      date: "2024-01-14",
      title: "Manipulated video of political leader's speech",
      status: "✅ Verified",
      statusColor: "bg-green-500/10 text-green-600"
    },
    {
      id: 3,
      category: "Finance",
      icon: "💰",
      date: "2024-01-13",
      title: "Investment scam promising guaranteed returns",
      status: "⚠️ Caution",
      statusColor: "bg-amber-500/10 text-amber-600"
    }
  ];

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
                Misinformation Archive
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-8">
                Browse verified cases of misinformation detection
              </p>
              
              {/* Stats Cards */}
              <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
                {stats.map((stat, index) => (
                  <div key={index} className="bg-card rounded-xl shadow-sm border border-border p-6">
                    <div className="text-3xl font-bold text-truthlens-primary mb-2">
                      {stat.number}
                    </div>
                    <div className="text-sm font-medium text-muted-foreground">
                      {stat.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Main Content */}
        <section className="py-8">
          <div className="container mx-auto px-4 lg:px-6">
            <div className="flex flex-col lg:flex-row gap-8">
              {/* Main Content Area */}
              <div className="flex-1">
                <Card className="mb-8">
                  <CardContent className="p-6">
                    {/* Search Bar */}
                    <div className="mb-6">
                      <Input 
                        placeholder="Search misinformation cases..."
                        className="w-full"
                      />
                    </div>
                    
                    {/* Filter Buttons */}
                    <div className="flex flex-wrap gap-3 mb-6">
                      {filters.map((filter) => (
                        <Button
                          key={filter}
                          variant={selectedFilter === filter ? "default" : "outline"}
                          size="sm"
                          onClick={() => setSelectedFilter(filter)}
                        >
                          {filter}
                        </Button>
                      ))}
                    </div>
                    
                    {/* Category Filters */}
                    <div className="flex flex-wrap gap-3">
                      {categories.map((category) => (
                        <Button
                          key={category.name}
                          variant={selectedCategory === category.name ? "default" : "ghost"}
                          size="sm"
                          onClick={() => setSelectedCategory(category.name)}
                        >
                          {category.icon} {category.name}
                        </Button>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Results Grid */}
                <div className="grid gap-6 mb-8">
                  {cases.map((case_) => (
                    <Card key={case_.id} className="hover:shadow-md transition-shadow">
                      <CardContent className="p-6">
                        <div className="flex items-start justify-between mb-4">
                          <div className="flex items-center gap-3">
                            <div className="text-2xl">{case_.icon}</div>
                            <div>
                              <div className="text-sm text-muted-foreground">{case_.date}</div>
                              <div className="font-medium text-card-foreground">{case_.category}</div>
                            </div>
                          </div>
                          <div className={`px-3 py-1 rounded-full text-sm font-medium ${case_.statusColor}`}>
                            {case_.status}
                          </div>
                        </div>
                        
                        <h3 className="text-lg font-semibold text-card-foreground mb-4">
                          {case_.title}
                        </h3>
                        
                        <Button variant="ghost" size="sm" className="text-primary hover:text-primary-hover">
                          View Details →
                        </Button>
                      </CardContent>
                    </Card>
                  ))}
                </div>

                <div className="text-center">
                  <Button variant="outline" size="lg">
                    Load More Cases
                  </Button>
                </div>
              </div>

              {/* Sidebar */}
              <div className="lg:w-80">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Contributing to the Database</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground mb-4">
                      Help us build a comprehensive database of misinformation cases by contributing your analysis and research.
                    </p>
                    <Button className="w-full">
                      Contribute Analysis →
                    </Button>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </section>
      </main>
      <TruthLensFooter />
    </div>
  );
};

export default Archive;