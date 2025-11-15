"""
LLM Configuration Analyzer: Use LLM to analyze system and suggest configurations
"""

import os
import json
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Run: pip install google-generativeai")


class LLMConfigAnalyzer:
    """Use LLM to analyze time series data and suggest Dainarx configurations"""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize LLM analyzer

        Args:
            model_name: Gemini model name (default from env: GEMINI_MODEL)
            api_key: Gemini API key (default from env: GEMINI_API_KEY)
        """
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")

        # Load environment variables
        load_dotenv()

        # Get API key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY not set. Please set it in .env file or pass as parameter."
            )

        # Get model name
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "models/gemini-flash-lite-latest")

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Initialize model
        self.model = genai.GenerativeModel(self.model_name)

        # Load prompt template
        self.prompt_template = self._load_prompt_template()

        print(f"LLM Analyzer initialized with model: {self.model_name}")

    def _load_prompt_template(self) -> str:
        """Load system analysis prompt template"""
        template_path = os.path.join(
            os.path.dirname(__file__),
            "prompts",
            "system_analysis.txt"
        )

        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                return f.read()
        else:
            # Fallback to basic template
            return """
You are an expert in hybrid systems. Analyze this time series data and suggest Dainarx configuration parameters.

{data_features}

Provide recommendations in JSON format with: order, other_items, window_size, kernel, svm_c, class_weight, self_loop, need_reset.
"""

    def analyze_system(self, data_features_text: str) -> Dict[str, Any]:
        """
        Analyze system characteristics and suggest configurations

        Args:
            data_features_text: Formatted data features string

        Returns:
            Dictionary containing:
                - recommendations: Dict of parameter suggestions with reasoning
                - system_type: Inferred system type
                - confidence: Confidence score (0-1)
        """
        # Format prompt
        prompt = self.prompt_template.format(data_features=data_features_text)

        print("Sending request to LLM...")
        print(f"Prompt length: {len(prompt)} characters")

        try:
            # Generate response
            response = self.model.generate_content(prompt)

            # Extract JSON from response
            suggestions = self._parse_json_response(response.text)

            print("LLM analysis completed successfully")
            return suggestions

        except Exception as e:
            print(f"Error calling LLM: {e}")
            # Return fallback conservative suggestions
            return self._get_fallback_suggestions()

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response (handles markdown code blocks)

        Args:
            text: Raw LLM response text

        Returns:
            Parsed JSON dictionary
        """
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in LLM response")

        # Parse JSON
        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"JSON string: {json_str[:200]}...")
            raise

    def _get_fallback_suggestions(self) -> Dict[str, Any]:
        """Get conservative fallback suggestions if LLM fails"""
        return {
            "recommendations": {
                "order": {
                    "value": [3, 4],
                    "reasoning": "Fallback: Conservative range for most systems"
                },
                "other_items": {
                    "value": ["", "x[?]**2", "x[?]**3"],
                    "reasoning": "Fallback: Test both linear and nonlinear"
                },
                "window_size": {
                    "value": [10, 12],
                    "reasoning": "Fallback: Standard range"
                },
                "kernel": {
                    "value": "rbf",
                    "reasoning": "Fallback: Most versatile kernel"
                },
                "svm_c": {
                    "value": [1e4, 1e6],
                    "reasoning": "Fallback: Standard range"
                },
                "class_weight": {
                    "value": [10, 30],
                    "reasoning": "Fallback: Moderate class balancing"
                },
                "self_loop": {
                    "value": True,
                    "reasoning": "Fallback: More conservative option"
                },
                "need_reset": {
                    "value": False,
                    "reasoning": "Fallback: Simpler model"
                }
            },
            "system_type": "unknown",
            "confidence": 0.0,
            "overall_reasoning": "Fallback suggestions due to LLM error"
        }

    def extract_config_ranges(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract configuration ranges from LLM response for genetic algorithm

        Args:
            llm_response: Response from analyze_system()

        Returns:
            Dictionary with ranges for each parameter
        """
        recommendations = llm_response.get("recommendations", {})

        config_ranges = {
            "order": self._ensure_list(recommendations.get("order", {}).get("value", [3])),
            "other_items": self._ensure_list(recommendations.get("other_items", {}).get("value", [""])),
            "window_size": self._ensure_list(recommendations.get("window_size", {}).get("value", [10])),
            "kernel": self._ensure_list(recommendations.get("kernel", {}).get("value", "rbf")),
            "svm_c": self._ensure_list(recommendations.get("svm_c", {}).get("value", [1e6])),
            "class_weight": self._ensure_list(recommendations.get("class_weight", {}).get("value", [10])),
            "self_loop": self._ensure_list(recommendations.get("self_loop", {}).get("value", False)),
            "need_reset": self._ensure_list(recommendations.get("need_reset", {}).get("value", False))
        }

        return config_ranges

    def _ensure_list(self, value):
        """Ensure value is a list"""
        if isinstance(value, list):
            return value
        else:
            return [value]

    def format_recommendations(self, llm_response: Dict[str, Any]) -> str:
        """Format LLM recommendations as readable text"""
        output = []
        output.append("=== LLM Configuration Analysis ===\n")

        output.append(f"System Type: {llm_response.get('system_type', 'unknown')}")
        output.append(f"Confidence: {llm_response.get('confidence', 0.0):.2f}\n")

        output.append(f"Overall Reasoning:\n{llm_response.get('overall_reasoning', 'N/A')}\n")

        output.append("=== Parameter Recommendations ===\n")

        recommendations = llm_response.get("recommendations", {})
        for param, info in recommendations.items():
            output.append(f"{param}:")
            output.append(f"  Value: {info.get('value')}")
            output.append(f"  Reasoning: {info.get('reasoning')}\n")

        return "\n".join(output)
