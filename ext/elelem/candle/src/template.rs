use anyhow::Result;
use minijinja::{context, Environment, Value};

pub struct Template {
    env: Environment<'static>,
}

impl Template {
    pub fn new(source: String) -> Result<Self> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("chat", source)?;
        Ok(Self { env })
    }

    pub fn render(
        &self,
        messages: &serde_json::Value,
        tools: &serde_json::Value,
        thinking: bool,
    ) -> Result<String> {
        let tools = if tools.as_array().is_some_and(|t| t.is_empty()) {
            &serde_json::Value::Null
        } else {
            tools
        };
        Ok(self.env.get_template("chat")?.render(context! {
            messages => Value::from_serialize(messages),
            tools => Value::from_serialize(tools),
            add_generation_prompt => true,
            enable_thinking => thinking,
        })?)
    }
}
