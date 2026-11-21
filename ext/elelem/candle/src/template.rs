use anyhow::Result;
use minijinja::value::Kwargs;
use minijinja::{context, Environment, Error, ErrorKind, Value};
use serde_json::ser::{Formatter, PrettyFormatter, Serializer};
use std::io;

pub struct Template {
    env: Environment<'static>,
}

impl Template {
    pub fn new(source: String) -> Result<Self> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
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

struct Python;

fn separate<W: ?Sized + io::Write>(w: &mut W, first: bool) -> io::Result<()> {
    if first {
        Ok(())
    } else {
        w.write_all(b", ")
    }
}

impl Formatter for Python {
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        separate(w, first)
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        separate(w, first)
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }
}

fn tojson(value: &Value, kwargs: Kwargs) -> Result<Value, Error> {
    let indent = kwargs.get::<Option<usize>>("indent")?;
    kwargs.assert_all_used()?;
    let mut out = Vec::new();
    let written = match indent {
        Some(n) => {
            let pad = vec![b' '; n];
            let mut ser = Serializer::with_formatter(&mut out, PrettyFormatter::with_indent(&pad));
            serde::Serialize::serialize(value, &mut ser)
        }
        None => {
            let mut ser = Serializer::with_formatter(&mut out, Python);
            serde::Serialize::serialize(value, &mut ser)
        }
    };
    written.map_err(invalid)?;
    String::from_utf8(out).map(Value::from).map_err(invalid)
}

fn invalid(e: impl std::fmt::Display) -> Error {
    Error::new(ErrorKind::InvalidOperation, e.to_string())
}
