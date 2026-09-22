// ==========================================================
// RDT-1.1 PRO WEBSITE
// ==========================================================

// CHANGE THIS AFTER DEPLOYING THE API.
//
// Example:
// https://your-render-service.onrender.com

const API_URL =
    "https://testrdt-6thc.onrender.com";


const form =
    document.getElementById("chat-form");

const promptInput =
    document.getElementById("prompt");

const chat =
    document.getElementById("chat");

const sendButton =
    document.getElementById("send");

const status =
    document.getElementById("status");


function addMessage(
    text,
    type
) {

    const message =
        document.createElement("div");

    message.className =
        "message " + type;

    message.textContent =
        text;

    chat.appendChild(
        message
    );

    chat.scrollTop =
        chat.scrollHeight;

    return message;
}


async function checkAPI() {

    try {

        const response =
            await fetch(
                API_URL + "/health"
            );

        if (!response.ok) {
            throw new Error(
                "API unavailable"
            );
        }

        const data =
            await response.json();

        status.textContent =
            "● RDT-1.1 Pro online";

    } catch (error) {

        status.textContent =
            "● API unavailable";
    }
}


form.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();

        const prompt =
            promptInput.value.trim();

        if (!prompt) {
            return;
        }

        addMessage(
            prompt,
            "message user"
        );

        promptInput.value = "";

        sendButton.disabled =
            true;

        sendButton.textContent =
            "Thinking...";

        const aiMessage =
            addMessage(
                "Generating...",
                "message ai"
            );

        try {

            const response =
                await fetch(
                    API_URL + "/generate",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            prompt:
                                "<|user|>\n"
                                + prompt
                                + "\n<|assistant|>\n",

                            max_new_tokens: 80,

                            temperature: 0.7,

                            top_k: 20
                        })
                    }
                );

            const data =
                await response.json();

            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Generation failed"
                );
            }

            aiMessage.textContent =
                data.response;

        } catch (error) {

            aiMessage.textContent =
                "Error: "
                + error.message;
        }

        sendButton.disabled =
            false;

        sendButton.textContent =
            "Send";
    }
);


checkAPI();
