import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import TruthLensHeader from "@/components/TruthLensHeader";
import TruthLensFooter from "@/components/TruthLensFooter";

const Authority = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const dashboardStats = [
    { icon: "📊", number: "1,247", label: "TOTAL" },
    { icon: "📅", number: "23", label: "TODAY'S" },
    { icon: "⏳", number: "15", label: "PENDING" },
    { icon: "🚨", number: "3", label: "HIGH PRIORITY" }
  ];

  const reports = [
    {
      id: 1,
      title: "Health misinformation campaign detected",
      category: "Health",
      priority: "High",
      date: "2024-01-15",
      status: "Under Review"
    },
    {
      id: 2,
      title: "Political deepfake video spreading",
      category: "Politics", 
      priority: "Critical",
      date: "2024-01-15",
      status: "Action Required"
    },
    {
      id: 3,
      title: "Financial scam targeting seniors",
      category: "Finance",
      priority: "Medium",
      date: "2024-01-14", 
      status: "Investigating"
    }
  ];

  const trendingCategories = [
    { name: "Health Misinformation", percentage: 67, color: "bg-accent-health" },
    { name: "Political Claims", percentage: 45, color: "bg-accent-political" },
    { name: "Financial Scams", percentage: 32, color: "bg-accent-finance" }
  ];

  if (!isLoggedIn) {
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
              <div className="max-w-2xl mx-auto">
                <div className="text-center mb-12">
                  <div className="text-6xl mb-6">🛡️</div>
                  <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-6">
                    Authority Access
                  </h1>
                  <p className="text-xl text-muted-foreground">
                    Real-time monitoring and reporting tools for verified authorities
                  </p>
                </div>

                <Card className="shadow-lg">
                  <CardContent className="p-8">
                    <div className="space-y-6">
                      <div>
                        <Label htmlFor="authority-id">Authority ID</Label>
                        <Input 
                          id="authority-id"
                          placeholder="Enter your authority identification"
                          className="mt-2"
                        />
                      </div>
                      <div>
                        <Label htmlFor="access-key">Access Key</Label>
                        <Input 
                          id="access-key"
                          type="password"
                          placeholder="Enter your secure access key"
                          className="mt-2"
                        />
                      </div>
                      <Button 
                        className="w-full"
                        size="lg"
                        onClick={() => setIsLoggedIn(true)}
                      >
                        Access Dashboard
                      </Button>
                    </div>
                  </CardContent>
                </Card>

                <Card className="mt-6 bg-secondary/20">
                  <CardHeader>
                    <CardTitle className="text-lg">Demo Credentials</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground mb-2">
                      <strong>Authority ID:</strong> DEMO_AUTH_2024
                    </p>
                    <p className="text-sm text-muted-foreground">
                      <strong>Access Key:</strong> demo_secure_key_123
                    </p>
                  </CardContent>
                </Card>
              </div>
            </div>
          </section>
        </main>
        <TruthLensFooter />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <TruthLensHeader />
      <main className="flex-1">
        {/* Dashboard Header */}
        <section className="bg-card border-b border-border">
          <div className="container mx-auto px-4 lg:px-6">
            <div className="flex items-center justify-between py-6">
              <h1 className="text-2xl font-bold text-foreground">Authority Dashboard</h1>
              <Button variant="outline" onClick={() => setIsLoggedIn(false)}>
                Logout
              </Button>
            </div>
          </div>
        </section>

        {/* Stats Grid */}
        <section className="py-8">
          <div className="container mx-auto px-4 lg:px-6">
            <div className="grid md:grid-cols-4 gap-6 mb-8">
              {dashboardStats.map((stat, index) => (
                <Card key={index}>
                  <CardContent className="p-6 text-center">
                    <div className="text-2xl mb-2">{stat.icon}</div>
                    <div className="text-2xl font-bold text-truthlens-primary mb-1">
                      {stat.number}
                    </div>
                    <div className="text-sm font-medium text-muted-foreground">
                      {stat.label}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            <div className="grid lg:grid-cols-3 gap-8">
              {/* Reports Table */}
              <div className="lg:col-span-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Recent Reports</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {reports.map((report) => (
                        <div 
                          key={report.id}
                          className="flex items-center justify-between p-4 bg-secondary/10 rounded-lg"
                        >
                          <div className="flex-1">
                            <h3 className="font-medium text-card-foreground mb-1">
                              {report.title}
                            </h3>
                            <div className="flex items-center gap-4 text-sm text-muted-foreground">
                              <span>{report.category}</span>
                              <span>•</span>
                              <span>{report.date}</span>
                              <span>•</span>
                              <span className={
                                report.priority === 'Critical' ? 'text-red-600' :
                                report.priority === 'High' ? 'text-amber-600' : 'text-green-600'
                              }>
                                {report.priority}
                              </span>
                            </div>
                          </div>
                          <div className="text-sm font-medium text-primary">
                            {report.status}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Trending Categories */}
              <div>
                <Card>
                  <CardHeader>
                    <CardTitle>Trending Categories</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {trendingCategories.map((category, index) => (
                      <div key={index}>
                        <div className="flex justify-between mb-2">
                          <span className="text-sm font-medium text-card-foreground">
                            {category.name}
                          </span>
                          <span className="text-sm text-muted-foreground">
                            {category.percentage}%
                          </span>
                        </div>
                        <div className="w-full bg-secondary/20 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full ${category.color}/70`}
                            style={{ width: `${category.percentage}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>

                <Card className="mt-6">
                  <CardHeader>
                    <CardTitle>Quick Actions</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Button className="w-full" variant="outline">
                      Generate Report
                    </Button>
                    <Button className="w-full" variant="outline">
                      Export Data
                    </Button>
                    <Button className="w-full">
                      Emergency Alert
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

export default Authority;