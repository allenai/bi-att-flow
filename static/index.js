(function() {

	var PNUM = 49
        var paragraph_id = 0;
	var titles;
	var contextss;
        var context_questions;

	window.onload = function(){
		//displayQuestion();
		load()
	}

	function clearField(){
		var qadiv = document.getElementById("qa");
		qadiv.innerHTML = "";
		displayQuestion();
		_loadQuestion();
	}

	function load(){
		sendAjax("/select", {}, handleParagraph);
	}
	function loadDropdown(){
		var dropdown = document.getElementById("selectArticle");
		for(var i=0; i<PNUM; i++){
			var opt = document.createElement("option");
			opt.value = parseInt(i);
			opt.innerHTML = titles[i];
			dropdown.appendChild(opt);
		}
		paragraph_id = 0;
		_loadParagraph();
		dropdown.onchange = loadParagraph;
	}

	function loadButtons(){
		var field = document.getElementById("button-field");
		for(var i=0; i<10; i++){
			var b = document.createElement("button");
			b.type = "button";
			b.classList.add("btn");
			b.classList.add("btn-sm");
			if (i%2==0) b.classList.add("btn-primary");
			else b.classList.add("btn-default");
			b.id = parseInt(i);
			b.innerHTML = "Para" + parseInt(i);
			b.onclick = loadParagraph;
			field.appendChild(b);
		}
	}

	function displayQuestion(){
		var form= document.createElement("div");
		form.id = "current";
		// label for Question
		var label = document.createElement("h4");
		var span = document.createElement("span");
		span.classList.add("label");
		span.classList.add("label-primary");
		span.innerHTML="Question";
		
		// Question List
		var q_select = document.createElement("select");
		q_select.name="selectQuestion";
		q_select.id="selectQuestion";
		
		// input for Qustion
		var input = document.createElement("input");
		input.type="text";
		input.name="question"
		input.classList.add("form-control");
		input.id="question";
		
		var breakdown = document.createElement("br");

		// button to submit
		var button = document.createElement("button");
		button.type = "button";
		button.classList.add("btn");
		button.classList.add("btn-sm");
		button.classList.add("btn-default");
		button.innerHTML="submit";
		button.id = "submit";
		// button to clear
		var clear = document.createElement("button");
		clear.type = "button";
		clear.classList.add("btn");
		clear.classList.add("btn-sm");
		clear.classList.add("btn-default");
		clear.innerHTML="clear";
		clear.id = "clear";
		// loading
		var loading = document.createElement("div");
		loading.id = "loading";
		loading.style.display = "none";
		var img = document.createElement("img");
		img.src = "https://webster.cs.washington.edu/images/babynames/loading.gif";
		img.alt = "icon";
		loading.appendChild(img);
		loading.innerHTML = loading.innerHTML + "loading";
		// appendChild
		form.appendChild(label);
		label.appendChild(span);
		if (paragraph_id !== 0)
			form.appendChild(q_select);
		form.appendChild(input);
		form.appendChild(breakdown);
		form.appendChild(button);
		form.appendChild(clear);
		form.appendChild(loading);
		var qadiv = document.getElementById("qa");
		qadiv.append(form);
		q_select.onchange = loadQuestion;
		button.onclick = loadAnswer;
		clear.onclick = clearField;
	}

	function displayAnswer(answer){
	        var div = document.createElement("div");
		var label = document.createElement("h4");
		var span = document.createElement("span");
		span.classList.add("label");
		span.classList.add("label-primary");
		span.innerHTML="Answer";
		var input = document.createElement("textarea");
		input.style = "resize:none";
		input.readOnly = true;
		input.classList.add("form-control");
		input.innerHTML = answer;
		div.appendChild(label);
		label.appendChild(span);
		div.appendChild(input);

		var qadiv = document.getElementById("qa");
		qadiv.append(div);
    }

     function loadAnswer(){
		document.getElementById("loading").style.display = "block";
		var data = {
			paragraph: $("#paragraph").val(),
			question: $("#question").val()
		};
		sendAjax("/submit", data, handleAnswer);
	}

    function handleAnswer(answer){
		var curr = document.getElementById("current");
		curr.removeChild(document.getElementById("submit"));
		curr.removeChild(document.getElementById("clear"));
		curr.removeChild(document.getElementById("loading"));
		displayAnswer(answer);
		var q = document.getElementById("question");
		q.id = "";
		q.readOnly = true;
		var q_select = document.getElementById("selectQuestion");
		if (q_select !== null)
			q_select.disabled = true;
		var clear = document.createElement("button");
		clear.type = "button";
		clear.classList.add("btn");
		clear.classList.add("btn-sm");
		clear.classList.add("btn-default");
		clear.innerHTML="new question!";
		clear.id = "clear";
		clear.onclick = clearField;
		curr.appendChild(clear);	
	}

	function loadQuestion(){
		var prev_q = document.getElementById("question");
		if (prev_q !== null){
			if (prev_q.name === "question")
				document.getElementById("current").removeChild(prev_q);
			else
				prev_q.id = "";
		}
		if (this.selectedIndex === 0) {
			// input for Qustion
			var input = document.createElement("input");
			input.type="text";
			input.name="question"
			input.classList.add("form-control");
			input.id="question";
			document.getElementById("current").insertBefore(input, document.getElementById("submit"));
		}
		else {
			var text = this.options[this.selectedIndex].text;
			this.options[this.selectedIndex].id = "question";
		}
	}

	function loadParagraph(){
		paragraph_id = this.value;
		_loadParagraph();
	}

	function _loadParagraph(){
		clearField();
		document.getElementById("paragraph").value = contextss[paragraph_id];
	}
	function _loadQuestion(){
		var questions = context_questions[paragraph_id];
		if (paragraph_id !== 0){
			var q_select = document.getElementById("selectQuestion");
			for (var i=0; i<questions.length+1; i++) {
				var opt = document.createElement("option");
				opt.name = parseInt(i);
				if (i === 0)
					opt.innerHTML = "Write own question";
				else
					opt.innerHTML = questions[i-1];
				opt.value = opt.innerHTML
				q_select.appendChild(opt);
			}
			q_select.onchange = loadQuestion;
		}
		// document.getElementById("question").value = context_questions[paragraph_id];
	}

	function handleParagraph(data){
		titles = data.titles;
		contextss = data.contextss;
                context_questions = data.context_questions;
		loadDropdown();
	}

        function sendAjax(url, data, handle){
		$.getJSON(url, data, function(response){
			handle(response.result);
		});
	}

})();
