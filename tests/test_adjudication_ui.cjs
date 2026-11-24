const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const template = fs.readFileSync(path.join(__dirname,
  "../data/validation/templates/adjudicate.html"), "utf8");
const script = template.match(/<script>\s*([\s\S]*?)<\/script>/)[1];

function reviewer() {
  const controls = { filter: { value: "all" }, fieldFilter: { value: "" }, find: { value: "" } };
  const context = vm.createContext({
    document: {
      getElementById: id => controls[id],
      createElement: () => ({
        dataset: {},
        querySelectorAll: () => [],
        querySelector: () => ({}),
      }),
    },
  });
  vm.runInContext(script.split('$("filter").onchange=render;')[0], context);
  vm.runInContext(`
    const makeCase=(doc,field,status)=>({
      case_id:doc+field,doc_id:doc,field,kind:"field",ratings:{},extraction:null,
      ai:{status,judgment:status==="proposed" ? "correct" : null,rationale:"Check the source.",evidence:[]}
    });
    data={package_id:"test",reviewer:"arno",documents:{
      W1:{title:"First document",pages:["Source"]},W2:{title:"Second document",pages:["Source"]}
    },cases:[
      makeCase("W1","query[10]","undecided"),makeCase("W1","query[2]","proposed"),
      makeCase("W1","query[1]","undecided"),makeCase("W2","objective","proposed")
    ]};
  `, context);
  return { context, controls };
}

test("proposals precede undecided cases across documents and filters", () => {
  const { context, controls } = reviewer();
  const fields = () => Array.from(vm.runInContext("filteredCases().map(item=>item.case_id)", context));
  assert.deepEqual(fields(), ["W1query[2]", "W2objective", "W1query[1]", "W1query[10]"]);
  controls.filter.value = "undecided";
  assert.deepEqual(fields(), ["W1query[1]", "W1query[10]"]);
  controls.filter.value = "proposed";
  assert.deepEqual(fields(), ["W1query[2]", "W2objective"]);
});

test("model assessment appears above case content and never finalizes a decision", () => {
  const { context } = reviewer();
  const proposed = vm.runInContext("renderCard(data.cases[1],0).innerHTML", context);
  const undecided = vm.runInContext("renderCard(data.cases[0],0).innerHTML", context);
  assert.ok(proposed.includes("GPT6.1 Medium — proposed resolution"));
  assert.ok(undecided.includes("GPT6.1 Medium — undecided"));
  assert.ok(proposed.indexOf("GPT6.1 Medium") < proposed.indexOf("Extracted value"));
  assert.ok(undecided.indexOf("GPT6.1 Medium") < undecided.indexOf("Extracted value"));
  assert.ok(proposed.includes("Provisional only. Your review is required."));
  assert.ok(proposed.includes("Use suggestion in draft"));
  assert.equal(vm.runInContext("review(data.cases[1]).judgment", context), null);
  assert.equal(vm.runInContext("review(data.cases[1]).review_status", context), "unreviewed");
});
